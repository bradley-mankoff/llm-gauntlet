"""Run a single benchmark against a single model. Used by run-gauntlet.sh."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_client import list_models, make_client  # noqa: E402
from tasks import codeeditbench, corebench, humaneval, ifeval, repoqa  # noqa: E402

TASKS = {
    "ifeval": ifeval,
    "codeeditbench": codeeditbench,
    "corebench": corebench,
    "humaneval": humaneval,
    "repoqa": repoqa,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=list(TASKS.keys()))
    ap.add_argument("--model", default="auto",
                    help="Model name as served by llama-server, or 'auto' to use the only model on the server")
    ap.add_argument("--base-url", default=os.environ.get("LLAMA_BASE_URL", "http://localhost:8080/v1"))
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42, help="Seed for random sampling (set to 0 to disable)")
    ap.add_argument("--think-off", action="store_true",
                    help="Inject <|think_off|> in user messages (froggeric hard switch; requires USE_FROGGERIC_CHAT_TEMPLATE=1 on the server)")
    ap.add_argument("--no-think", action="store_true",
                    help="Append /no_think to user messages (Qwen soft switch; weaker than --think-off)")
    ap.add_argument("--out", required=True, help="Output JSON file")
    args = ap.parse_args()

    client = make_client(base_url=args.base_url)
    models = list_models(client)
    print(f"[bench_one] server reports models: {models}")

    if args.model == "auto":
        if len(models) != 1:
            print(f"ERROR: server has {len(models)} models, can't auto-pick", file=sys.stderr)
            return 2
        args.model = models[0]
    elif args.model not in models:
        print(f"WARNING: requested model '{args.model}' not in server's list {models}", file=sys.stderr)

    print(f"[bench_one] benchmark={args.benchmark} model={args.model} base_url={args.base_url}")

    task = TASKS[args.benchmark]
    kwargs = {}
    if args.n_samples is not None:
        kwargs["n_samples"] = args.n_samples
    if args.max_tokens is not None:
        kwargs["max_tokens"] = args.max_tokens
    if args.seed:
        kwargs["seed"] = args.seed
    if args.think_off:
        kwargs["think_off"] = True
    if args.no_think:
        kwargs["no_think"] = True

    t0 = time.time()
    summary, results = task.run(client, args.model, **kwargs)
    summary["wall_time_sec"] = round(time.time() - t0, 1)
    summary["base_url"] = args.base_url

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\n[bench_one] done in {summary['wall_time_sec']}s")
    print(json.dumps(summary, indent=2))
    print(f"[bench_one] saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
