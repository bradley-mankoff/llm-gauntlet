"""HumanEval: implement a function from a docstring. 164 problems, execution-graded.

Uses the `openai/openai_humaneval` dataset from HuggingFace. Grading is
execution-based: the model's completion is concatenated with the problem's
`test` block and run; pass means the test's `check(<entry_point>)` runs
without AssertionError or other exception.

The "firm plan" your input described is exactly this format:
  - Input:  a function signature + docstring (the spec / plan)
  - Output: a working implementation
  - Grade:  real test cases that the model never sees

Run: uv run python bench_one.py --benchmark humaneval --model auto \
     --n-samples 164 --max-tokens 1024 --seed 42
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from datasets import load_dataset
from bench_client import chat as _bench_chat  # noqa: E402

NAME = "humaneval"
MAX_TOKENS_DEFAULT = 2048
TEST_TIMEOUT_SEC = 10
DATASET_ID = "openai/openai_humaneval"
CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.+?)\n```", re.DOTALL)
DEF_LINE_RE_TEMPLATE = r"def\s+{}\s*\("


def _strip_code_fence(text: str) -> str:
    """If the completion is wrapped in a markdown code fence, return the inner code."""
    m = CODE_FENCE_RE.search(text)
    if m:
        return m.group(1).rstrip()
    return text


def _extract_function(completion: str, entry_point: str) -> str | None:
    """Find `def <entry_point>(` anywhere in the completion and return the code
    from that line onwards. Returns None if no function definition is found
    (caller will treat the whole completion as the body to append to the prompt)."""
    pattern = DEF_LINE_RE_TEMPLATE.format(re.escape(entry_point))
    match = re.search(pattern, completion)
    return completion[match.start():] if match else None


def _grade(prompt: str, completion: str, test_code: str, entry_point: str) -> tuple[bool, str | None]:
    """Run the model's code + the test cases in a subprocess. Returns (passed, error)."""
    code_body = _strip_code_fence(completion)
    # The model often wraps the function in prose ("Here's the implementation: ...").
    # If we can find `def <entry_point>(` in the completion, use just that function
    # (don't double up with the prompt's signature). Otherwise treat as the body.
    function_def = _extract_function(code_body, entry_point)
    if function_def is not None:
        full_code = function_def + "\n" + test_code
    else:
        full_code = prompt + "\n" + code_body + "\n" + test_code
    full_code += f"\ncheck({entry_point})\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT_SEC,
        )
        if result.returncode == 0:
            return True, None
        return False, (result.stderr or result.stdout or "")[:500]
    except subprocess.TimeoutExpired:
        return False, f"timeout after {TEST_TIMEOUT_SEC}s"
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def run(client, model: str, n_samples: int | None = None, max_tokens: int = MAX_TOKENS_DEFAULT,
        temperature: float = 0.0, progress_every: int = 1, seed: int | None = 42, **kwargs):
    ds = load_dataset(DATASET_ID, split="test", cache_dir="./.cache")
    if seed:
        ds = ds.shuffle(seed=seed)
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))

    print(f"[humaneval] {len(ds)} samples, model={model}, max_tokens={max_tokens}, seed={seed}, test_timeout={TEST_TIMEOUT_SEC}s")

    results: list[dict] = []
    for i, row in enumerate(ds):
        task_id = row["task_id"]
        prompt = row["prompt"]
        test_code = row["test"]
        entry_point = row["entry_point"]

        try:
            resp = _bench_chat(
                client,
                model,
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                think_off=bool(kwargs.get("think_off", False)),
                no_think=bool(kwargs.get("no_think", False)),
            )
            completion = (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            results.append({"task_id": task_id, "passed": False, "error": repr(exc)})
            continue

        passed, err = _grade(prompt, completion, test_code, entry_point)
        results.append({
            "task_id": task_id,
            "passed": passed,
            "completion_len": len(completion),
            "completion_preview": completion[:500],
            "error": (err[:200] if err else None),
        })

        if (i + 1) % progress_every == 0 or i == 0:
            p = sum(1 for r in results if r.get("passed"))
            n = len(results)
            print(f"  [{i + 1}/{len(ds)}] pass={p}/{n} ({p / n:.2%})")

    passed = sum(1 for r in results if r.get("passed"))
    summary = {
        "benchmark": NAME,
        "model": model,
        "n_samples": len(results),
        "passed": passed,
        "pass_rate": passed / len(results) if results else 0.0,
        "seed": seed,
    }
    return summary, results
