"""Scout benchmark runner — standalone (separate from bench_one.py).

Usage:
    uv run python bench_scout.py \
        --model auto --base-url http://localhost:8000/v1 \
        --repo ~/llama-runs/gauntlet \
        --queries queries/scout_gauntlet.json \
        --out results/scout_gauntlet.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_client import list_models, make_client
from tasks.scout import run as scout_run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="auto")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--repo", required=True, help="Path to codebase to search")
    ap.add_argument("--queries", required=True, help="JSON file with queries array")
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--out", required=True, help="Output JSON file")
    args = ap.parse_args()

    client = make_client(base_url=args.base_url)
    models = list_models(client)

    if args.model == "auto":
        if len(models) != 1:
            print(f"ERROR: server has {len(models)} models, can't auto-pick", file=sys.stderr)
            return 2
        args.model = models[0]

    with open(args.queries) as f:
        queries = json.load(f)

    print(f"[scout] model={args.model} repo={args.repo} queries={len(queries)}")

    t0 = time.time()
    summary, results = scout_run(
        client, args.model,
        repo_path=args.repo,
        queries=queries,
        n_samples=args.n_samples,
        max_tokens=args.max_tokens,
    )
    summary["wall_time_sec"] = round(time.time() - t0, 1)
    summary["base_url"] = args.base_url

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\n[scout] done in {summary['wall_time_sec']}s")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
