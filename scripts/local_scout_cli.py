#!/usr/bin/env python3
"""CLI for OMP agents: candidate list + file windows + optional tiered walk."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tiered_scout import (  # noqa: E402
    candidates_json,
    file_window,
    list_candidates,
    make_openai_judge,
    run_tiered_scout,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Local tiered scout CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("candidates", help="Ranked file candidates (retrieval only)")
    p_list.add_argument("--repo", required=True)
    p_list.add_argument("--query", required=True)
    p_list.add_argument("--top-n", type=int, default=5)
    p_list.add_argument("--graphify", action="store_true")

    p_win = sub.add_parser("window", help="Numbered window for one file")
    p_win.add_argument("--repo", required=True)
    p_win.add_argument("--file", required=True)
    p_win.add_argument("--start-line", type=int, default=1)
    p_win.add_argument("--max-lines", type=int, default=160)

    p_run = sub.add_parser("walk", help="Full tiered walk with a judge OpenAI-compatible endpoint")
    p_run.add_argument("--repo", required=True)
    p_run.add_argument("--query", required=True)
    p_run.add_argument("--top-n", type=int, default=5)
    p_run.add_argument("--max-peeks", type=int, default=5)
    p_run.add_argument("--graphify", action="store_true")
    p_run.add_argument("--base-url", default=os.environ.get("JUDGE_BASE_URL", "http://127.0.0.1:8080/v1"))
    p_run.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "auto"))
    p_run.add_argument("--api-key", default=os.environ.get("JUDGE_API_KEY", "sk-noop"))
    p_run.add_argument("--think-off", action="store_true")
    p_run.add_argument("--no-think", action="store_true")

    args = ap.parse_args()

    if args.cmd == "candidates":
        cands = list_candidates(args.repo, args.query, top_n=args.top_n, use_graphify=args.graphify)
        print(candidates_json(cands))
        return 0

    if args.cmd == "window":
        print(file_window(args.repo, args.file, start_line=args.start_line, max_lines=args.max_lines))
        return 0

    if args.cmd == "walk":
        model = args.model
        if model == "auto":
            from openai import OpenAI

            client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=60)
            models = client.models.list().data
            if not models:
                print(json.dumps({"status": "error", "reason": "no models on judge endpoint"}))
                return 2
            model = models[0].id
        judge = make_openai_judge(
            base_url=args.base_url,
            model=model,
            api_key=args.api_key,
            think_off=args.think_off,
            no_think=args.no_think,
        )
        cap = run_tiered_scout(
            args.repo,
            args.query,
            judge,
            top_n=args.top_n,
            max_peeks=args.max_peeks,
            use_graphify=args.graphify,
        )
        print(json.dumps(cap.to_dict(), indent=2))
        return 0 if cap.status in {"accepted", "exhausted"} else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
