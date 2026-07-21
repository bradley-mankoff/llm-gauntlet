#!/usr/bin/env python3
"""Run scout + executor benchmarks on new models. Assumes server is running.

Usage:
  # Serve the model first:
  BENCH=1 MODEL_PRESET=minicpm5-v2 ./scripts/serve.sh

  # Then run benchmarks:
  uv run python bench_new_models.py --preset minicpm5-v2

  # Or just:
  uv run python bench_new_models.py --model auto --base-url http://localhost:8080/v1

Models each get a results/pipeline_<preset>.json, results/humaneval_<preset>.json,
and results/corebench_<preset>.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
QUERIES_FILE = REPO_ROOT / "queries" / "scout_mtplx.json"
REPO_PATH = None  # Set to the codebase path for scout pipeline, or auto-detect

PRESETS = {
    "minicpm5-v2": {
        "froggeric": False,
        "think_off": False,
        "description": "MiniCPM5 1B v2 thinking-on (Q8_0)",
    },
    "qwythos-v2": {
        "froggeric": False,
        "think_off": False,  # Qwythos has its own enable_thinking control
        "description": "Qwythos 9B v2 (Q4_K_M)",
    },
    "thinkingcap": {
        "froggeric": True,
        "think_off": False,
        "description": "ThinkingCap 27B thinking-on (Q4_K_M)",
    },
    "qwen4": {
        "froggeric": True,
        "think_off": True,
        "description": "Qwen3.6 27B think-off (Q4_K_XL)",
    },
}


def run_benchmark(benchmark: str, model: str, base_url: str, out: str,
                  extra_args: list[str] | None = None) -> dict | None:
    """Run bench_one.py for a single benchmark. Returns summary dict."""
    cmd = [
        sys.executable, "-m", "uv", "run", "python",
        str(REPO_ROOT / "bench_one.py"),
        "--benchmark", benchmark,
        "--model", model,
        "--base-url", base_url,
        "--out", str(REPO_ROOT / "results" / out),
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"[runner] {benchmark} on {model}")
    print(f"[runner] {' '.join(cmd)}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(
        ["uv", "run", "python", str(REPO_ROOT / "bench_one.py"),
         "--benchmark", benchmark,
         "--model", model,
         "--base-url", base_url,
         "--out", str(REPO_ROOT / "results" / out),
         *(extra_args or [])],
        cwd=str(REPO_ROOT),
        capture_output=False,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"[runner] {benchmark} FAILED (exit {result.returncode}, {elapsed:.0f}s)")
        return None

    print(f"[runner] {benchmark} done ({elapsed:.0f}s)")

    # Read summary from output file
    out_path = REPO_ROOT / "results" / out
    if out_path.exists():
        with open(out_path) as f:
            data = json.load(f)
        return data.get("summary", {})
    return None


def run_pipeline(model: str, base_url: str, out: str, repo: str,
                 n_samples: int | None = None) -> dict | None:
    """Run bench_pipeline.py for scout benchmark."""
    cmd = [
        "uv", "run", "python", str(REPO_ROOT / "bench_pipeline.py"),
        "--model", model,
        "--base-url", base_url,
        "--repo", repo,
        "--queries", str(QUERIES_FILE),
        "--out", str(REPO_ROOT / "results" / out),
    ]
    if n_samples:
        cmd.extend(["--n-samples", str(n_samples)])

    print(f"\n{'='*60}")
    print(f"[runner] scout_pipeline on {model}")
    print(f"[runner] {' '.join(cmd)}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"[runner] pipeline FAILED (exit {result.returncode}, {elapsed:.0f}s)")
        return None

    print(f"[runner] pipeline done ({elapsed:.0f}s)")

    out_path = REPO_ROOT / "results" / out
    if out_path.exists():
        with open(out_path) as f:
            data = json.load(f)
        return data.get("summary", {})
    return None


def main():
    ap = argparse.ArgumentParser(description="Run benchmarks on new models")
    ap.add_argument("--model", default="auto",
                    help="Model name (auto = detect from server)")
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--preset", choices=list(PRESETS),
                    help="Preset name (sets think_off/froggeric flags)")
    ap.add_argument("--repo", help="Codebase path for scout pipeline")
    ap.add_argument("--n-samples", type=int, default=None,
                    help="Limit samples per benchmark")
    ap.add_argument("--benchmarks", nargs="+",
                    default=["humaneval", "corebench"],
                    choices=["humaneval", "ifeval", "corebench", "repoqa"],
                    help="Which benchmarks to run (default: humaneval corebench)")
    ap.add_argument("--skip-pipeline", action="store_true",
                    help="Skip scout pipeline benchmark")
    ap.add_argument("--pipeline-only", action="store_true",
                    help="Only run scout pipeline")
    args = ap.parse_args()

    extra_args = []
    if args.preset:
        preset = PRESETS[args.preset]
        if preset["think_off"]:
            extra_args.append("--think-off")
        print(f"[runner] preset={args.preset}: {preset['description']}")

    # Auto-detect model if needed
    model_arg = args.model

    results_summary = {}

    # Scout pipeline
    if not args.skip_pipeline:
        repo = args.repo
        if not repo:
            print("[runner] ERROR: --repo required for scout pipeline", file=sys.stderr)
            print("[runner] Example: --repo ~/path/to/MTPLX", file=sys.stderr)
            return 1

        n_pipeline = min(args.n_samples, 30) if args.n_samples else None
        out_name = f"pipeline_{args.preset or 'model'}.json"
        summary = run_pipeline(model_arg, args.base_url, out_name, repo, n_pipeline)
        results_summary["pipeline"] = summary

    if not args.pipeline_only:
        for bench in args.benchmarks:
            bench_extra = list(extra_args)
            out_name = f"{bench}_{args.preset or 'model'}.json"

            # Bench-specific args
            if bench == "humaneval":
                if args.n_samples:
                    bench_extra.extend(["--n-samples", str(min(args.n_samples, 50))])
                else:
                    bench_extra.extend(["--n-samples", "50"])
            elif bench == "corebench":
                if args.n_samples:
                    bench_extra.extend(["--n-samples", str(min(args.n_samples, 5))])
                else:
                    bench_extra.extend(["--n-samples", "5"])
                bench_extra.extend(["--max-tokens", "4096"])
            elif bench == "ifeval":
                if args.n_samples:
                    bench_extra.extend(["--n-samples", str(min(args.n_samples, 100))])
                else:
                    bench_extra.extend(["--n-samples", "50"])
                bench_extra.extend(["--max-tokens", "8192"])

            summary = run_benchmark(bench, model_arg, args.base_url, out_name, bench_extra)
            results_summary[bench] = summary

    # Print summary
    print(f"\n{'='*60}")
    print("[runner] SUMMARY")
    print(f"{'='*60}")
    for name, summary in results_summary.items():
        if summary:
            print(f"\n{name}:")
            for k, v in summary.items():
                print(f"  {k}: {v}")
        else:
            print(f"\n{name}: FAILED or not run")

    return 0


if __name__ == "__main__":
    sys.exit(main())
