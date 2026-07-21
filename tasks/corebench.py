"""CORE-Bench (easy): read scientific code output and answer questions.

Uses the CORE-Bench train set (siegelz/core-bench). Each task provides a
scientific code repository with pre-computed results. The model must read
the results directory and answer specific questions — pure code-reading
comprehension, no dependency installation or code execution.

Level: codeocean_easy — results directory is preserved, model just reads.

Capsules are downloaded from corebench.cs.princeton.edu and cached locally.
"""
from __future__ import annotations

import json
import os
import re
import tarfile
import time
import urllib.request
from pathlib import Path

from bench_client import chat as _bench_chat

NAME = "corebench"
MAX_TOKENS_DEFAULT = 4096
CAPSULE_BASE_URL = "https://corebench.cs.princeton.edu/capsules"
CACHE_DIR = Path(".cache/corebench")
TRAIN_DATASET = Path("../core-bench/benchmark/dataset/core_train.json")

# Tolerance for numeric answer comparison
NUMERIC_TOLERANCE = 0.01


def _download_capsule(capsule_id: str) -> Path:
    """Download and extract a capsule, cached to .cache/corebench/."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    capsule_dir = CACHE_DIR / capsule_id

    if capsule_dir.exists():
        return capsule_dir

    url = f"{CAPSULE_BASE_URL}/{capsule_id}.tar.gz"
    tar_path = CACHE_DIR / f"{capsule_id}.tar.gz"

    print(f"  [corebench] downloading {url} ...")
    urllib.request.urlretrieve(url, tar_path)

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=CACHE_DIR)

    tar_path.unlink()
    return capsule_dir


def _read_results_dir(capsule_dir: Path) -> dict[str, str]:
    """Read all files in the results/ directory, return {filename: content}.

    Capsules have results in capsule_dir/results/ (most common), or
    occasionally nested deeper. We walk the tree and collect any file
    that isn't obviously code or metadata.
    """
    results: dict[str, str] = {}
    skip_patterns = [
        re.compile(r"\.(pyc?|r|R|sh|md|txt|json|yml|yaml|cfg|ini|toml)$"),
        re.compile(r"^(README|REPRODUCING|Dockerfile|Makefile|setup\.)", re.I),
    ]

    for root, _, files in os.walk(capsule_dir):
        for fname in files:
            fpath = Path(root) / fname
            rel = str(fpath.relative_to(capsule_dir))

            # Only collect files that look like results (not code/config)
            is_code_or_config = any(p.search(fname) for p in skip_patterns)
            if is_code_or_config or fname.startswith("."):
                continue

            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
                # Skip empty or very large files
                if 0 < len(content) < 100_000:
                    results[rel] = content
            except Exception:
                pass

    # Also explicitly include common result-bearing files
    for glob_pat in ["results/**/*", "**/results/**/*", "**/*.csv", "**/*.tsv", "**/*.log", "**/*.out"]:
        for fpath in capsule_dir.glob(glob_pat):
            if fpath.is_file() and str(fpath.relative_to(capsule_dir)) not in results:
                try:
                    content = fpath.read_text(encoding="utf-8", errors="replace")
                    if 0 < len(content) < 100_000:
                        results[str(fpath.relative_to(capsule_dir))] = content
                except Exception:
                    pass

    return results


def _build_prompt(task: dict, results: dict[str, str]) -> str:
    """Build a single prompt for the LLM: task description + files + questions."""
    questions = list(task["results"][0].keys())
    question_list = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

    file_sections = []
    for fname, content in sorted(results.items()):
        truncated = content[:5000]  # cap per file
        file_sections.append(f"### File: {fname}\n```\n{truncated}\n```")

    files_block = "\n\n".join(file_sections) if file_sections else "(no result files found)"

    return (
        f"You are analyzing the output of scientific code from the paper "
        f"\"{task['capsule_title']}\".\n\n"
        f"Task: {task['task_prompt']}\n\n"
        f"Below are the contents of the results directory. Read them carefully "
        f"and answer each question. Return ONLY a valid JSON object mapping each "
        f"question to its answer. For numeric answers, use the exact number "
        f"(not a string).\n\n"
        f"Questions:\n{question_list}\n\n"
        f"Files:\n{files_block}\n\n"
        f"Return your answer as a JSON object with these exact keys:\n"
        f"{json.dumps(questions)}"
    )


def _parse_response(response: str, questions: list[str]) -> dict | None:
    """Extract JSON from the LLM response. Returns dict or None on failure."""
    # Try direct JSON parse first
    text = response.strip()
    try:
        result = json.loads(text)
        if isinstance(result, dict) and all(q in result for q in questions):
            return result
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code fence
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(1))
            if isinstance(result, dict) and any(q in result for q in questions):
                return result
        except json.JSONDecodeError:
            pass

    # Try to find a bare JSON object
    m = re.search(r"\{[^{}]*?\}", text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(0))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return None


def _grade(reported: dict, ground_truths: list[dict]) -> tuple[int, int]:
    """Compare reported answers to ground truth. Returns (correct, total).

    Numeric answers are compared with tolerance. String answers use
    case-insensitive substring match. List answers use exact match.
    """
    gt = ground_truths[0]  # use first ground truth entry
    total = len(gt)
    correct = 0

    for question, gt_value in gt.items():
        if question not in reported:
            continue
        reported_value = reported[question]

        if isinstance(gt_value, (int, float)):
            try:
                rv = float(reported_value)
                gv = float(gt_value)
                if gv == 0:
                    if abs(rv) < NUMERIC_TOLERANCE:
                        correct += 1
                elif abs(rv - gv) / abs(gv) < NUMERIC_TOLERANCE:
                    correct += 1
            except (ValueError, TypeError):
                pass
        elif isinstance(gt_value, list):
            if reported_value == gt_value:
                correct += 1
        elif isinstance(gt_value, str):
            if str(reported_value).lower().strip() == gt_value.lower().strip():
                correct += 1

    return correct, total


def run(client, model: str, n_samples: int | None = None,
        max_tokens: int = MAX_TOKENS_DEFAULT,
        temperature: float = 0.0, progress_every: int = 1,
        seed: int | None = 42, **kwargs):
    """Run CORE-Bench easy against the train set.

    Returns (summary, results) matching the bench_one contract.
    """
    # Load dataset
    ds_path = Path(TRAIN_DATASET).resolve()
    if not ds_path.exists():
        raise FileNotFoundError(
            f"CORE-Bench dataset not found at {ds_path}. "
            f"Clone https://github.com/siegelz/core-bench to ../core-bench"
        )

    with open(ds_path) as f:
        tasks = json.load(f)

    if seed:
        import random
        rng = random.Random(seed)
        rng.shuffle(tasks)

    if n_samples:
        tasks = tasks[:n_samples]

    print(f"[corebench] {len(tasks)} tasks, model={model}, max_tokens={max_tokens}")
    if seed:
        print(f"[corebench] seed={seed}, shuffled")

    results: list[dict] = []
    total_correct = 0
    total_questions = 0
    total_time = 0.0
    parse_failures = 0

    for i, task in enumerate(tasks):
        capsule_id = task["capsule_id"]
        questions = list(task["results"][0].keys())

        # Download and extract capsule
        capsule_dir = _download_capsule(capsule_id)

        # Read results
        result_files = _read_results_dir(capsule_dir)

        # Build prompt
        prompt = _build_prompt(task, result_files)

        # Call LLM
        t0 = time.time()
        try:
            resp = _bench_chat(
                client, model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            elapsed = time.time() - t0
            response_text = resp.choices[0].message.content or ""
        except Exception as e:
            elapsed = time.time() - t0
            results.append({
                "capsule_id": capsule_id,
                "capsule_title": task["capsule_title"],
                "error": str(e),
                "elapsed_s": round(elapsed, 1),
                "correct": 0,
                "total": len(questions),
                "task": task,
            })
            continue

        # Parse response
        reported = _parse_response(response_text, questions)

        if reported is None:
            parse_failures += 1
            results.append({
                "capsule_id": capsule_id,
                "capsule_title": task["capsule_title"],
                "response": response_text[:500],
                "parse_error": True,
                "elapsed_s": round(elapsed, 1),
                "correct": 0,
                "total": len(questions),
                "task": task,
            })
            if (i + 1) % progress_every == 0:
                print(f"  [{i+1}/{len(tasks)}] {capsule_id} — parse failure ({elapsed:.0f}s)")
            continue

        # Grade
        correct, total = _grade(reported, task["results"])
        total_correct += correct
        total_questions += total
        total_time += elapsed

        results.append({
            "capsule_id": capsule_id,
            "capsule_title": task["capsule_title"],
            "reported": reported,
            "ground_truth": task["results"][0],
            "correct": correct,
            "total": total,
            "elapsed_s": round(elapsed, 1),
            "task": task,
        })

        if (i + 1) % progress_every == 0:
            print(f"  [{i+1}/{len(tasks)}] {capsule_id} — "
                  f"{correct}/{total} correct ({elapsed:.0f}s)")

    n = len(tasks)
    summary = {
        "benchmark": NAME,
        "model": model,
        "n_tasks": n,
        "total_questions": total_questions,
        "total_correct": total_correct,
        "accuracy": round(total_correct / total_questions, 4) if total_questions else 0,
        "parse_failures": parse_failures,
        "avg_time_s": round(total_time / n, 1) if n else 0,
        "total_time_s": round(total_time, 1),
        "max_tokens": max_tokens,
        "seed": seed,
    }

    return summary, results
