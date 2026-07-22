"""Tiered scout: retrieval top-k files, then one-file-at-a-time line capsule.

Stage A — local retrieval (embedder + reranker [+ optional graphify])
Stage B — judge model inspects ONE candidate file at a time (minimal context)
         until it ACCEPTS with line range + code, or list exhausted.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from scout_pipeline import ScoutPipeline, _extract_code_block, _parse_file_ref


@dataclass
class Candidate:
    file: str
    symbol: str = ""
    start_line: int = 1
    rerank_score: float | None = None
    source: str = "embeddings"
    preview: str = ""


@dataclass
class Capsule:
    status: str  # accepted | exhausted | error
    query: str
    file: str | None = None
    symbol: str | None = None
    lines: str | None = None  # "start-end"
    code: str = ""
    reason: str = ""
    tried: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)
    peeks: int = 0
    elapsed_s: float = 0.0
    raw_judge: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



class UsageMeter:
    """Accumulate prompt/completion tokens from judge LLM calls."""

    def __init__(self) -> None:
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.calls: int = 0

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += int(prompt or 0)
        self.completion_tokens += int(completion or 0)
        self.calls += 1

    @property
    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "calls": self.calls,
        }


def _norm(p: str) -> str:
    return str(Path(str(p))).replace("\\", "/").lstrip("./")


def list_candidates(
    repo: str | Path,
    query: str,
    *,
    top_n: int = 5,
    top_k: int = 20,
    use_graphify: bool = False,
    collection_name: str = "scout_code",
    pipeline: ScoutPipeline | None = None,
) -> list[Candidate]:
    pipe = pipeline or ScoutPipeline(str(repo), collection_name=collection_name)
    chunks = pipe.retrieve(
        query,
        top_k=top_k,
        rerank_top_n=top_n,
        use_graphify=use_graphify,
    )
    out: list[Candidate] = []
    seen: set[str] = set()
    for c in chunks:
        f = c.get("file") or ""
        if not f or f in seen:
            continue
        seen.add(f)
        score = c.get("rerank_score")
        out.append(
            Candidate(
                file=f,
                symbol=str(c.get("symbol") or ""),
                start_line=int(c.get("start_line") or 1),
                rerank_score=float(score) if score is not None else None,
                source=str(c.get("source") or "embeddings"),
                preview=str(c.get("text") or "")[:900],
            )
        )
    return out


def file_window(
    repo: str | Path,
    rel_file: str,
    *,
    start_line: int = 1,
    context_before: int = 15,
    max_lines: int = 160,
    max_chars: int = 6000,
) -> str:
    path = Path(repo) / rel_file
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return f"[error reading {rel_file}: {exc}]"
    if not lines:
        return "[empty]"
    lo = max(0, int(start_line) - 1 - context_before)
    hi = min(len(lines), lo + max_lines)
    numbered = [f"{lo + i + 1:>6}|{row}" for i, row in enumerate(lines[lo:hi])]
    text = f"// FILE: {rel_file} lines {lo + 1}-{lo + len(numbered)} of {len(lines)}\n" + "\n".join(numbered)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return text


def build_judge_prompt(
    query: str,
    candidate: Candidate,
    window: str,
    peek_i: int,
    peek_n: int,
    feedback: str | None = None,
) -> str:
    fb = (feedback or "").strip()
    feedback_section = ""
    if fb:
        feedback_section = (
            "\n\nOrchestrator feedback from prior attempts (do not ignore):\n"
            f"{fb}\n"
        )
    return f"""You are the line-scout judge. You see ONE window of ONE candidate file.

Query:
{query}
{feedback_section}
Candidate {peek_i}/{peek_n}: {candidate.file}
symbol_hint: {candidate.symbol or "?"}
rerank_score: {candidate.rerank_score if candidate.rerank_score is not None else "n/a"}

File window:
{window}

Decide one of:
1) ACCEPT — this window contains the definition site for the query.
2) MORE — likely the right file, but definition is outside this window (need another region).
3) REJECT — wrong file / wrong layer / only a mention or facade.

If ACCEPT, output exactly:
VERDICT: ACCEPT
FILE: {candidate.file}
SYMBOL: <name or unknown>
LINES: <start>-<end>
REASON: <one short sentence>

If MORE, output exactly:
VERDICT: MORE
CENTER_LINE: <line number to center next window on>
SYMBOL: <symbol to seek if known>
REASON: <one short sentence>

If REJECT, output exactly:
VERDICT: REJECT
REASON: <one short sentence>

Rules:
- Prefer definition bodies over facades/re-exports.
- LINES must use the numbered lines in the window.
- Do NOT paste a large code block; LINES are enough (system re-reads source).
- Do not invent paths.
- Use MORE instead of ACCEPT if you only see a forward declare, call site, or unrelated sibling.
"""



_VERDICT_RE = re.compile(r"VERDICT:\s*(ACCEPT|REJECT|MORE)", re.I)
_LINES_RE = re.compile(r"LINES:\s*(\d+)\s*-\s*(\d+)", re.I)
_SYMBOL_RE = re.compile(r"SYMBOL:\s*(\S+)", re.I)
_REASON_RE = re.compile(r"REASON:\s*(.+)", re.I)
_CENTER_RE = re.compile(r"CENTER_LINE:\s*(\d+)", re.I)


def parse_judge(raw: str, fallback_file: str) -> dict[str, Any]:
    text = raw or ""
    vm = _VERDICT_RE.search(text)
    verdict = (vm.group(1).upper() if vm else "")
    if not verdict:
        # heuristic: if model emitted FILE+LINES treat as accept
        if _LINES_RE.search(text) and (_parse_file_ref(text) or fallback_file):
            verdict = "ACCEPT"
        else:
            verdict = "REJECT"
    file_ref = _parse_file_ref(text, candidates=[fallback_file]) or fallback_file
    lm = _LINES_RE.search(text)
    lines = f"{lm.group(1)}-{lm.group(2)}" if lm else None
    sm = _SYMBOL_RE.search(text)
    symbol = sm.group(1) if sm else None
    rm = _REASON_RE.search(text)
    reason = (rm.group(1).strip() if rm else "")
    cm = _CENTER_RE.search(text)
    center_line = int(cm.group(1)) if cm else None
    code = _extract_code_block(text) if verdict == "ACCEPT" else ""
    # if extract_code_block returns whole text, trim when no fences
    if code.strip() == text.strip() and "```" not in text:
        code = ""
    return {
        "verdict": verdict,
        "file": file_ref,
        "lines": lines,
        "symbol": symbol,
        "reason": reason,
        "code": code,
        "center_line": center_line,
    }


JudgeFn = Callable[[str], str]


def _query_keyword_anchors(path: Path, query: str, *, limit: int = 6) -> list[int]:
    """Find lines in file matching salient query tokens (helps when retrieval symbol is wrong)."""
    if not path.is_file():
        return []
    stop = {
        "find", "code", "that", "which", "with", "from", "this", "that", "into", "when",
        "where", "what", "function", "method", "class", "return", "returns", "using",
        "after", "before", "about", "should", "would", "could", "their", "there", "have",
        "file", "line", "lines", "block", "implementation", "define", "defined",
    }
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]{4,}", query or "")
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", query or "")
    scored: list[tuple[int, str]] = []
    seen: set[str] = set()
    for t in toks:
        tl = t.lower()
        if tl in stop or tl in seen or len(tl) < 3:
            continue
        seen.add(tl)
        score = len(tl)
        if "_" in tl or any(c.isupper() for c in tl[1:]):
            score += 5
        scored.append((score, tl))
    # Add derived variants: first 4 chars (e.g. "initializes" -> "init")
    extra: list[tuple[int, str]] = []
    for _, t in scored:
        if len(t) >= 7:
            for span in [t[:4], t[1:5], t[-4:]]:
                if span not in seen and span not in stop:
                    seen.add(span)
                    extra.append((4, span))
    scored.extend(extra)
    scored.sort(reverse=True)
    keys = [t for _, t in scored[:12]]
    if not keys:
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    hits: list[int] = []
    for i, ln in enumerate(lines, 1):
        s = ln.strip()
        if not s or s.startswith(("//", "/*", "*", "#include")):
            continue
        for k in keys:
            if k in ln:
                hits.append(i)
                break
        if len(hits) >= limit * 3:
            break
    # diversify by spacing
    out: list[int] = []
    for h in hits:
        if all(abs(h - o) > 40 for o in out):
            out.append(h)
        if len(out) >= limit:
            break
    return out

def walk_candidates(
    repo: str | Path,
    query: str,
    candidates: list[Candidate],
    judge: JudgeFn,
    *,
    max_peeks: int = 5,
    max_within_file: int = 3,
    feedback: str | None = None,
) -> Capsule:
    from scout_metrics import find_symbol_lines, parse_line_range, slice_file_lines

    t0 = time.time()
    tried: list[str] = []
    last_raw = ""
    peeks = 0
    n = min(len(candidates), max_peeks)
    repo = Path(repo)

    for i, cand in enumerate(candidates[:n], 1):
        tried.append(cand.file)
        path = repo / cand.file

        # Anchors: retrieval start + symbol hits + query-keyword hits.
        anchors: list[int] = []
        if cand.start_line and int(cand.start_line) > 0:
            anchors.append(int(cand.start_line))
        for h in find_symbol_lines(path, cand.symbol or "", limit=5):
            if all(abs(h - a) > 25 for a in anchors):
                anchors.append(h)
        for h in _query_keyword_anchors(path, query, limit=4):
            if all(abs(h - a) > 25 for a in anchors):
                anchors.append(h)
        if not anchors:
            anchors = [1]
        within = 0
        seen_centers: set[int] = set()
        queue = list(anchors[: max(2, min(4, max_within_file))])

        while queue and within < max_within_file:
            center = queue.pop(0)
            bucket = int(round(center / 30.0) * 30) or center
            if bucket in seen_centers:
                continue
            seen_centers.add(bucket)
            within += 1
            peeks += 1

            window = file_window(repo, cand.file, start_line=center)
            prompt = build_judge_prompt(query, cand, window, i, n, feedback=feedback)
            raw = judge(prompt)
            last_raw = raw or ""
            parsed = parse_judge(last_raw, cand.file)
            verdict = parsed["verdict"]

            if verdict == "MORE":
                nxt = parsed.get("center_line")
                if not nxt and parsed.get("symbol"):
                    hits = find_symbol_lines(path, parsed["symbol"], limit=3)
                    nxt = hits[0] if hits else None
                if nxt and within < max_within_file:
                    queue.append(int(nxt))
                continue

            if verdict == "ACCEPT":
                lines = parsed["lines"]
                rng = parse_line_range(lines)
                code = ""
                if rng:
                    code = slice_file_lines(path, rng[0], rng[1])
                if not code:
                    code = parsed.get("code") or ""
                return Capsule(
                    status="accepted",
                    query=query,
                    file=parsed["file"] or cand.file,
                    symbol=parsed["symbol"] or cand.symbol or None,
                    lines=lines,
                    code=code,
                    reason=parsed["reason"],
                    tried=tried,
                    candidates=[c.file for c in candidates],
                    peeks=peeks,
                    elapsed_s=round(time.time() - t0, 2),
                    raw_judge=last_raw[:1000],
                )

            # REJECT -> next candidate file
            break

    return Capsule(
        status="exhausted",
        query=query,
        reason="no candidate accepted",
        tried=tried,
        candidates=[c.file for c in candidates],
        peeks=peeks,
        elapsed_s=round(time.time() - t0, 2),
        raw_judge=last_raw[:1000],
    )


def make_openai_judge(
    *,
    base_url: str,
    model: str,
    api_key: str = "sk-noop",
    max_tokens: int = 450,
    temperature: float = 0.0,
    timeout: float = 900.0,
    think_off: bool = False,
    no_think: bool = False,
    extra_body: dict | None = None,
    meter: UsageMeter | None = None,
) -> JudgeFn:
    from openai import OpenAI

    # local import to reuse message_text when available
    try:
        from bench_client import message_text
    except Exception:
        def message_text(message) -> str:  # type: ignore
            return getattr(message, "content", None) or ""

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    body = dict(extra_body or {})

    def _judge(prompt: str) -> str:
        import time as _time

        content = prompt
        if think_off and "<|think_off|>" not in content:
            content = f"<|think_off|>\n{content}"
        if no_think and "/no_think" not in content:
            content = content + "\n/no_think"

        last_err: Exception | None = None
        for attempt in range(12):
            try:
                kwargs: dict = {
                    "model": model,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if body:
                    kwargs["extra_body"] = body
                resp = client.chat.completions.create(**kwargs)
                if meter is not None:
                    usage = getattr(resp, "usage", None)
                    if usage is not None:
                        meter.add(
                            getattr(usage, "prompt_tokens", 0) or 0,
                            getattr(usage, "completion_tokens", 0) or 0,
                        )
                if not resp.choices:
                    return ""
                return message_text(resp.choices[0].message)
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
                retryable = status in {408, 409, 429, 500, 502, 503, 504} or any(
                    s in msg for s in ("rate-limited", "rate limit", "429", "temporar", "overloaded", "timeout")
                )
                if not retryable or attempt >= 11:
                    raise
                # free-tier upstream limits need long sleeps
                delay = min(120.0, (2 ** attempt) * 1.5)
                _time.sleep(delay)
        if last_err:
            raise last_err
        return ""

    return _judge


def make_codex_judge(
    *,
    model: str = "gpt-5.6-luna",
    api_key: str | None = None,
    reasoning_effort: str = "high",
    timeout: float = 900.0,
    base_url: str = "https://chatgpt.com/backend-api",
    instructions: str = "You are a precise code scout judge. Follow the output format exactly.",
    meter: UsageMeter | None = None,
) -> JudgeFn:
    """Judge via OpenAI Codex ChatGPT backend (SSE Responses API)."""
    import base64
    import json as _json
    import time as _time
    import urllib.error
    import urllib.request

    token = (api_key or "").strip()
    if not token:
        raise ValueError("codex judge needs api_key/token")

    try:
        payload = _json.loads(base64.urlsafe_b64decode(token.split(".")[1] + "=="))
        account_id = payload["https://api.openai.com/auth"]["chatgpt_account_id"]
    except Exception as e:
        raise ValueError(f"codex token missing chatgpt_account_id: {e}") from e

    raw = base_url.rstrip("/")
    if raw.endswith("/codex/responses"):
        url = raw
    elif raw.endswith("/codex"):
        url = raw + "/responses"
    else:
        url = raw + "/codex/responses"

    def _judge(prompt: str) -> str:
        body = {
            "model": model,
            "store": False,
            "stream": True,
            "instructions": instructions,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
            "text": {"verbosity": "low"},
            "include": ["reasoning.encrypted_content"],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "reasoning": {"effort": reasoning_effort, "summary": "auto"},
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "OpenAI-Beta": "responses=experimental",
            "chatgpt-account-id": account_id,
            "originator": "pi",
            "User-Agent": "pi (darwin)",
        }
        data = _json.dumps(body).encode()
        last_err: Exception | None = None
        for attempt in range(10):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    texts: list[str] = []
                    final = ""
                    buf = ""
                    while True:
                        chunk = resp.read(4096)
                        if not chunk:
                            break
                        buf += chunk.decode("utf-8", "replace")
                        while "\n" in buf:
                            line, buf = buf.split("\n", 1)
                            line = line.strip("\r")
                            if not line.startswith("data: "):
                                continue
                            payload_s = line[6:]
                            if payload_s == "[DONE]":
                                continue
                            try:
                                ev = _json.loads(payload_s)
                            except Exception:
                                continue
                            et = ev.get("type") or ""
                            if et == "response.output_text.delta":
                                texts.append(ev.get("delta") or "")
                            elif et == "response.output_text.done":
                                if ev.get("text"):
                                    final = ev.get("text") or ""
                            elif et == "response.failed":
                                raise RuntimeError(str(ev)[:500])
                            elif meter is not None and et == "response.completed":
                                try:
                                    resp_obj = ev.get("response") or ev
                                    usage = (
                                        resp_obj.get("usage")
                                        if isinstance(resp_obj, dict)
                                        else None
                                    )
                                    if isinstance(usage, dict):
                                        prompt_t = (
                                            usage.get("input_tokens")
                                            if usage.get("input_tokens") is not None
                                            else usage.get("prompt_tokens")
                                        )
                                        completion_t = (
                                            usage.get("output_tokens")
                                            if usage.get("output_tokens") is not None
                                            else usage.get("completion_tokens")
                                        )
                                        if prompt_t is not None or completion_t is not None:
                                            meter.add(prompt_t or 0, completion_t or 0)
                                except Exception:
                                    pass
                    return (final or "".join(texts)).strip()
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                code = getattr(e, "code", None)
                if hasattr(e, "read"):
                    try:
                        msg = (e.read() or b"").decode("utf-8", "replace").lower() + " " + msg
                    except Exception:
                        pass
                retryable = code in {408, 409, 429, 500, 502, 503, 504} or any(
                    s in msg for s in ("rate", "429", "temporar", "overloaded", "timeout", "stream")
                )
                if not retryable or attempt >= 9:
                    raise
                _time.sleep(min(90.0, (2**attempt) * 1.25))
        if last_err:
            raise last_err
        return ""

    return _judge



def run_tiered_scout(
    repo: str | Path,
    query: str,
    judge: JudgeFn,
    *,
    top_n: int = 5,
    max_peeks: int = 5,
    use_graphify: bool = False,
    pipeline: ScoutPipeline | None = None,
    feedback: str | None = None,
) -> Capsule:
    cands = list_candidates(
        repo,
        query,
        top_n=top_n,
        use_graphify=use_graphify,
        pipeline=pipeline,
    )
    if not cands:
        return Capsule(status="error", query=query, reason="no candidates")
    return walk_candidates(
        repo, query, cands, judge, max_peeks=max_peeks, feedback=feedback
    )


def candidates_json(cands: list[Candidate]) -> str:
    return json.dumps([asdict(c) for c in cands], indent=2)
