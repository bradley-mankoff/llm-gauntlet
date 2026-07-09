"""RepoQA: long-context code retrieval. The "scout" gauntlet slot.

Grading per task: extract the first ```...``` code block from the model
response, compute sentence-BLEU against the needle function, pass if BLEU
>= threshold (default 0.8, matching the paper).

Notes:
  - Data is auto-downloaded from the evalplus/repoqa_release GitHub release
    on first use (~70 MB).
  - The vendored `tree_sitter` extraction in the upstream package is not
    available on Python 3.14; we use a regex-based code-block extractor
    instead. This is slightly looser (a model that wraps its answer in
    a single ``` block gets full credit, regardless of syntactic validity).
  - Prompt template matches the upstream `search_needle_function.py`:
        instruction + code_context + description + instruction
  - For the gauntlet we use the full file content as the code context
    rather than the paper's 16k-token topologically-sorted slice. That's
    a v0 simplification — the file is usually <30k tokens and well within
    llama.cpp's 256k ctx budget.
"""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

NAME = "repoqa"
MAX_TOKENS_DEFAULT = 1024
BLEU_THRESHOLD_DEFAULT = 0.8
DATA_PATH = Path(__file__).resolve().parent.parent / ".cache" / "repoqa-2024-06-23.json"
DATA_URL = "https://github.com/evalplus/repoqa_release/releases/download/2024-06-23/repoqa-2024-06-23.json.gz"

INSTRUCTION = (
    "Based on the function description and code context, "
    "please retrieve and repeat the exact described function from the code context "
    "in a code block wrapped by ```:"
)


def _ensure_data() -> Path:
    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 1_000_000:
        return DATA_PATH
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[repoqa] downloading dataset from {DATA_URL}")
    import gzip
    import io
    with urllib.request.urlopen(DATA_URL) as resp:
        raw = resp.read()
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
        DATA_PATH.write_bytes(gz.read())
    print(f"[repoqa] saved to {DATA_PATH} ({DATA_PATH.stat().st_size:,} bytes)")
    return DATA_PATH


def _load_data() -> dict:
    path = _ensure_data()
    with open(path) as f:
        return json.load(f)


def _flatten_tasks(data: dict) -> list[dict]:
    """Flatten {lang: [{repo, needles: [...]}]} into a list of task dicts."""
    out: list[dict] = []
    for lang, repos in data.items():
        for repo_entry in repos:
            content_map = repo_entry.get("content", {})
            for needle in repo_entry.get("needles", []):
                out.append({
                    "lang": lang,
                    "repo": repo_entry.get("repo", "?"),
                    "topic": repo_entry.get("topic", "?"),
                    "path": needle["path"],
                    "needle_name": needle["name"],
                    "start_line": needle["start_line"],
                    "end_line": needle["end_line"],
                    "description": needle.get("description", ""),
                    "code_context": content_map.get(needle["path"], ""),
                })
    return out


_CODE_BLOCK_RE = re.compile(r"```(?:\w+)?[ \t]*\n(.*?)\n```", re.DOTALL)


def _extract_code_block(response: str) -> str:
    m = _CODE_BLOCK_RE.search(response)
    if m:
        return m.group(1).strip()
    return response.strip()


def _bleu(reference: str, hypothesis: str) -> float:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    ref_tokens, hyp_tokens = reference.split(), hypothesis.split()
    if not hyp_tokens or not ref_tokens:
        return 0.0
    return sentence_bleu(
        [ref_tokens], hyp_tokens,
        smoothing_function=SmoothingFunction().method1,
    )


def _extract_needle_function(content: str, start_line: int, end_line: int) -> str:
    lines = content.split("\n")
    return "\n".join(lines[start_line:end_line + 1])


def run(client, model: str, n_samples: int | None = None, max_tokens: int = MAX_TOKENS_DEFAULT,
        temperature: float = 0.0, bleu_threshold: float = BLEU_THRESHOLD_DEFAULT,
        progress_every: int = 1, **kwargs):
    data = _load_data()
    tasks = _flatten_tasks(data)
    if n_samples is not None:
        tasks = tasks[:n_samples]
    print(f"[repoqa] {len(tasks)} tasks, model={model}, max_tokens={max_tokens}, bleu_threshold={bleu_threshold}")

    results: list[dict] = []
    for i, task in enumerate(tasks):
        needle_func = _extract_needle_function(task["code_context"], task["start_line"], task["end_line"])
        prompt = (
            f"{INSTRUCTION}\n\n"
            f"{task['code_context']}\n\n"
            f"{task['description']}\n\n"
            f"{INSTRUCTION}"
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            response_text = (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            results.append({"task_idx": i, "lang": task["lang"], "repo": task["repo"],
                            "needle": task["needle_name"], "error": repr(exc), "pass": False})
            continue

        extracted = _extract_code_block(response_text)
        score = _bleu(needle_func, extracted)
        passed = score >= bleu_threshold

        results.append({
            "task_idx": i,
            "lang": task["lang"],
            "repo": task["repo"],
            "path": task["path"],
            "needle": task["needle_name"],
            "response": response_text,
            "extracted": extracted[:600],
            "needle_func_len": len(needle_func),
            "bleu": round(score, 4),
            "pass": passed,
        })

        if (i + 1) % progress_every == 0 or i == 0:
            p = sum(1 for r in results if r.get("pass"))
            n = len(results)
            avg_bleu = sum(r.get("bleu", 0) for r in results) / n if n else 0
            print(f"  [{i + 1}/{len(tasks)}] pass={p}/{n} ({p / n:.2%})  avg_bleu={avg_bleu:.3f}")

    passed = sum(1 for r in results if r.get("pass"))
    avg_bleu = (sum(r.get("bleu", 0) for r in results) / len(results)) if results else 0
    summary = {
        "benchmark": NAME,
        "model": model,
        "n_tasks": len(tasks),
        "passed": passed,
        "pass_rate": passed / len(tasks) if tasks else 0.0,
        "avg_bleu": round(avg_bleu, 4),
        "bleu_threshold": bleu_threshold,
    }
    return summary, results
