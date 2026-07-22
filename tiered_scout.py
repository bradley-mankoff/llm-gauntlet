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


def build_judge_prompt(query: str, candidate: Candidate, window: str, peek_i: int, peek_n: int) -> str:
    return f"""You are the line-scout judge. You see ONE candidate file at a time.

Query:
{query}

Candidate {peek_i}/{peek_n}: {candidate.file}
symbol_hint: {candidate.symbol or "?"}
rerank_score: {candidate.rerank_score if candidate.rerank_score is not None else "n/a"}

File window:
{window}

Decide:
- ACCEPT if this file is the definition site for the query behavior (not a thin wrapper/re-export/hub facade).
- REJECT if wrong file, wrong layer, or only mentions the behavior.

If ACCEPT, output exactly:
VERDICT: ACCEPT
FILE: {candidate.file}
SYMBOL: <name or unknown>
LINES: <start>-<end>
```language
<minimal code excerpt that answers the query>
```
REASON: <one short sentence>

If REJECT, output exactly:
VERDICT: REJECT
REASON: <one short sentence>

Rules:
- Prefer definition bodies over facades.
- LINES must refer to the numbered window above.
- Keep the code excerpt minimal but sufficient.
- Do not invent paths.
"""


_VERDICT_RE = re.compile(r"VERDICT:\s*(ACCEPT|REJECT)", re.I)
_LINES_RE = re.compile(r"LINES:\s*(\d+)\s*-\s*(\d+)", re.I)
_SYMBOL_RE = re.compile(r"SYMBOL:\s*(\S+)", re.I)
_REASON_RE = re.compile(r"REASON:\s*(.+)", re.I)


def parse_judge(raw: str, fallback_file: str) -> dict[str, Any]:
    text = raw or ""
    vm = _VERDICT_RE.search(text)
    verdict = (vm.group(1).upper() if vm else "")
    if not verdict:
        # heuristic: if model emitted FILE+LINES treat as accept
        if _LINES_RE.search(text) and ( _parse_file_ref(text) or fallback_file):
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
    }


JudgeFn = Callable[[str], str]


def walk_candidates(
    repo: str | Path,
    query: str,
    candidates: list[Candidate],
    judge: JudgeFn,
    *,
    max_peeks: int = 5,
) -> Capsule:
    t0 = time.time()
    tried: list[str] = []
    last_raw = ""
    peeks = 0
    n = min(len(candidates), max_peeks)
    for i, cand in enumerate(candidates[:n], 1):
        peeks += 1
        tried.append(cand.file)
        window = file_window(repo, cand.file, start_line=cand.start_line)
        prompt = build_judge_prompt(query, cand, window, i, n)
        raw = judge(prompt)
        last_raw = raw or ""
        parsed = parse_judge(last_raw, cand.file)
        if parsed["verdict"] == "ACCEPT":
            return Capsule(
                status="accepted",
                query=query,
                file=parsed["file"] or cand.file,
                symbol=parsed["symbol"] or cand.symbol or None,
                lines=parsed["lines"],
                code=parsed["code"] or "",
                reason=parsed["reason"],
                tried=tried,
                candidates=[c.file for c in candidates],
                peeks=peeks,
                elapsed_s=round(time.time() - t0, 2),
                raw_judge=last_raw[:1000],
            )
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
    max_tokens: int = 700,
    temperature: float = 0.0,
    timeout: float = 180.0,
    think_off: bool = False,
    no_think: bool = False,
) -> JudgeFn:
    from openai import OpenAI

    # local import to reuse message_text when available
    try:
        from bench_client import message_text
    except Exception:
        def message_text(message) -> str:  # type: ignore
            return getattr(message, "content", None) or ""

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def _judge(prompt: str) -> str:
        content = prompt
        if think_off and "<|think_off|>" not in content:
            content = f"<|think_off|>\n{content}"
        if no_think and "/no_think" not in content:
            content = content + "\n/no_think"
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if not resp.choices:
            return ""
        return message_text(resp.choices[0].message)

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
    return walk_candidates(repo, query, cands, judge, max_peeks=max_peeks)


def candidates_json(cands: list[Candidate]) -> str:
    return json.dumps([asdict(c) for c in cands], indent=2)
