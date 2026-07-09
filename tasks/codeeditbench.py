"""CodeEditBench: directive-driven code edits. The "plan implementation" gauntlet slot.

For v0 we support the `code_debug_primary` split (debugging) — the simplest
of the four task types (debug / translate / polish / switch). The data is
auto-downloaded from `m-a-p/CodeEditorBench` on first use (~160 MB).

Grading: simple BLEU between the model's response and the reference
`solutions[0]`. This is a v0 approximation — the upstream benchmark uses
execution-based grading (run the hidden tests). For a real run you'll
want to swap in the upstream `evaluation/` grader, but for model-vs-model
ranking the BLEU surrogate correlates well.

Prompt format (debug):
    There is a bug in the following code. Please find and fix it.
    Return only the fixed code in a code block wrapped by ```.

    ```<lang>
    <source_code>
    ```
"""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

NAME = "codeeditbench"
MAX_TOKENS_DEFAULT = 4096
DATA_DIR = Path(__file__).resolve().parent.parent / ".cache" / "codeeditbench"
HF_BASE = "https://huggingface.co/datasets/m-a-p/CodeEditorBench/resolve/main"
DEFAULT_SPLIT = "code_debug_primary.jsonl"
BLEU_THRESHOLD_DEFAULT = 0.8

PROMPTS = {
    "code_debug_primary": (
        "There is a bug in the following code. Please find and fix it. "
        "Return only the fixed code in a code block wrapped by ```."
    ),
    "code_translate_primary": (
        "Translate the following code from {src_lang} to {tgt_lang}. "
        "Return only the translated code in a code block wrapped by ```."
    ),
    "code_polishment_primary": (
        "Polish the following code to improve readability, performance, and idiomatic style. "
        "Return only the polished code in a code block wrapped by ```."
    ),
    "code_switch_primary": (
        "Modify the following code to satisfy the new requirement. "
        "Return only the modified code in a code block wrapped by ```."
    ),
}


def _download_split(split: str = DEFAULT_SPLIT) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / split
    if out.exists() and out.stat().st_size > 100_000:
        return out
    print(f"[codeeditbench] downloading {split} from {HF_BASE}/{split}")
    urllib.request.urlretrieve(f"{HF_BASE}/{split}", out)
    print(f"[codeeditbench] saved to {out} ({out.stat().st_size:,} bytes)")
    return out


def _load_split(split: str = DEFAULT_SPLIT) -> list[dict]:
    path = _download_split(split)
    samples: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


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


def _build_prompt(sample: dict, split: str) -> str:
    base = PROMPTS.get(split, PROMPTS["code_debug_primary"])
    code_lang = sample.get("code_language") or sample.get("source_lang") or "python"
    source = sample.get("source_code") or sample.get("incorrect_solutions", "")
    return f"{base}\n\n```{code_lang}\n{source}\n```"


def _reference_solution(sample: dict) -> str:
    sols = sample.get("solutions", "")
    if isinstance(sols, str) and sols:
        # Sometimes solutions is a JSON-stringified list
        try:
            parsed = json.loads(sols)
            if isinstance(parsed, list) and parsed:
                return parsed[0]
        except json.JSONDecodeError:
            return sols
        return sols
    return ""


import random  # noqa: E402


def run(client, model: str, n_samples: int | None = None, max_tokens: int = MAX_TOKENS_DEFAULT,
        temperature: float = 0.0, bleu_threshold: float = BLEU_THRESHOLD_DEFAULT,
        split: str = DEFAULT_SPLIT, seed: int | None = 42, progress_every: int = 1, **kwargs):
    samples = _load_split(split)
    if seed:
        rng = random.Random(seed)
        rng.shuffle(samples)
    if n_samples is not None:
        samples = samples[:n_samples]
    print(f"[codeeditbench] {len(samples)} samples (split={split}), model={model}, max_tokens={max_tokens}")

    results: list[dict] = []
    for i, sample in enumerate(samples):
        prompt = _build_prompt(sample, split)
        reference = _reference_solution(sample)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            response_text = (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            results.append({"task_idx": i, "split": split, "error": repr(exc), "pass": False})
            continue

        extracted = _extract_code_block(response_text)
        score = _bleu(reference, extracted) if reference else 0.0
        passed = bool(reference) and score >= bleu_threshold

        results.append({
            "task_idx": i,
            "split": split,
            "title": sample.get("title", ""),
            "difficulty": sample.get("difficulty", ""),
            "code_lang": sample.get("code_language") or sample.get("source_lang", ""),
            "response": response_text,
            "extracted": extracted[:600],
            "reference_len": len(reference),
            "bleu": round(score, 4),
            "pass": passed,
        })

        if (i + 1) % progress_every == 0 or i == 0:
            p = sum(1 for r in results if r.get("pass"))
            n = len(results)
            avg_bleu = sum(r.get("bleu", 0) for r in results) / n if n else 0
            print(f"  [{i + 1}/{len(samples)}] pass={p}/{n} ({p / n:.2%})  avg_bleu={avg_bleu:.3f}")

    passed = sum(1 for r in results if r.get("pass"))
    avg_bleu = (sum(r.get("bleu", 0) for r in results) / len(results)) if results else 0
    summary = {
        "benchmark": NAME,
        "model": model,
        "split": split,
        "n_samples": len(samples),
        "passed": passed,
        "pass_rate": passed / len(samples) if samples else 0.0,
        "avg_bleu": round(avg_bleu, 4),
        "bleu_threshold": bleu_threshold,
    }
    return summary, results
