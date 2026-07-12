"""Scout benchmark: code retrieval from a codebase.

Simulates the "scout" role: embedder -> reranker -> scout LLM.
Context capped at a few files to simulate realistic post-reranker input.
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

from bench_client import chat as _bench_chat

NAME = "scout"
MAX_TOKENS_DEFAULT = 4096
MAX_CONTEXT_FILES = 5


def _build_context(repo_path: str, target_files: list[str]) -> str:
    """Build context from specific files."""
    repo = Path(repo_path).expanduser().resolve()
    parts = []
    for rel_path in target_files:
        f = repo / rel_path
        if not f.exists():
            continue
        try:
            content = f.read_text()
        except Exception:
            continue
        parts.append(f"// FILE: {rel_path}\n{content}")
    return "\n\n".join(parts)


def _extract_code_block(response: str) -> str:
    m = re.search(r"```(?:\w+)?\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return response.strip()


def _parse_file_ref(response: str) -> str | None:
    m = re.search(r"(?:FILE:\s*|in\s+)([\w/\-_.]+\.\w+)", response, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"`([\w/\-_.]+\.\w+)`", response)
    if m:
        return m.group(1)
    return None


def _build_prompt(query: str, context: str) -> str:
    return textwrap.dedent(f"""\
    You are a code scout. Given a codebase and a query, find the exact location
    of the relevant code. Return:
    1. The file path
    2. The function/class/symbol name
    3. The exact code (copy-pasted from the file)

    Respond in this format:

    FILE: path/to/file.py
    SYMBOL: function_name
    ```language
    [exact code from the file]
    ```

    ## Codebase

    {context}

    ## Query

    {query}
    """)


def run(client, model: str, repo_path: str, queries: list[dict],
        n_samples: int | None = None, max_tokens: int = MAX_TOKENS_DEFAULT,
        temperature: float = 0.0, progress_every: int = 1, **kwargs):
    repo = Path(repo_path).expanduser().resolve()
    all_py = sorted([
        str(p.relative_to(repo))
        for p in repo.glob("**/*.py")
        if p.is_file() and ".venv" not in str(p) and "__pycache__" not in str(p)
    ])[:MAX_CONTEXT_FILES]

    context = _build_context(str(repo), all_py)
    print(f"[scout] {len(all_py)} files, {len(context)} chars context")

    if n_samples:
        queries = queries[:n_samples]

    results = []
    passed = 0
    file_matches = 0
    total_bleu = 0.0

    print(f"[scout] {len(queries)} queries, model={model}")

    for i, q in enumerate(queries):
        prompt = _build_prompt(q["query"], context)
        resp = _bench_chat(client, model, [{"role": "user", "content": prompt}],
                           max_tokens=max_tokens, temperature=temperature, **kwargs)
        response_text = resp.choices[0].message.content
        extracted = _extract_code_block(response_text)
        file_ref = _parse_file_ref(response_text)

        gt_file = q["file"]
        gt_code = q["code"]

        file_ok = file_ref is not None and Path(file_ref).name == Path(gt_file).name

        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            bleu = sentence_bleu([gt_code.split()], extracted.split(),
                                 smoothing_function=SmoothingFunction().method1)
        except Exception:
            bleu = 1.0 if extracted.strip() == gt_code.strip() else 0.0

        code_ok = bleu >= 0.8
        if code_ok:
            passed += 1
        if file_ok:
            file_matches += 1
        total_bleu += bleu

        results.append({
            "query": q["query"],
            "gt_file": gt_file,
            "gt_symbol": q.get("symbol", ""),
            "response": response_text[:500],
            "extracted_code": extracted[:500],
            "file_ref": file_ref,
            "file_ok": file_ok,
            "bleu": round(bleu, 4),
            "code_ok": code_ok,
        })

        if (i + 1) % progress_every == 0:
            pf = file_matches / (i + 1)
            pc = passed / (i + 1)
            print(f"  [{i+1}/{len(queries)}] file={pf:.1%} code={pc:.1%}")

    n = len(queries)
    summary = {
        "benchmark": NAME,
        "model": model,
        "n_samples": n,
        "file_match_rate": round(file_matches / n, 4) if n else 0,
        "code_match_rate": round(passed / n, 4) if n else 0,
        "avg_bleu": round(total_bleu / n, 4) if n else 0,
    }
    return summary, results
