#!/usr/bin/env python3
"""Benchmark tiered scout (retrieval top-k + sequential judge) with file + code metrics."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bench_pipeline import (  # noqa: E402
    _file_hit,
    _gold_files_for,
    _rank_in_candidates,
    _write_output,
)
from scout_metrics import rebuild_query_gold, score_capsule  # noqa: E402
from scout_pipeline import ScoutPipeline  # noqa: E402
from tiered_scout import make_codex_judge, make_openai_judge, run_tiered_scout  # noqa: E402




def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--max-peeks", type=int, default=5)
    ap.add_argument("--graphify", action="store_true")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="auto")
    ap.add_argument("--api-key", default="sk-noop")
    ap.add_argument("--think-off", action="store_true")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        help="OpenRouter/OpenAI-style reasoning.effort for thinking models",
    )
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument(
        "--judge-backend",
        default="openai",
        choices=["openai", "codex"],
        help="openai=chat completions; codex=ChatGPT Codex Responses SSE",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    if args.n_samples:
        queries = queries[: args.n_samples]

    out_path = Path(args.out)
    results: list[dict] = []
    if args.resume and out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            results = list(prev.get("results") or [])
        except Exception:
            results = []

    model = args.model
    if args.judge_backend == "codex":
        if model == "auto":
            model = "gpt-5.6-luna"
        effort = args.reasoning_effort or "high"
        judge = make_codex_judge(
            model=model,
            api_key=args.api_key,
            reasoning_effort=effort,
            timeout=args.timeout,
            base_url=args.base_url if "chatgpt.com" in args.base_url or args.base_url.endswith("/codex") else "https://chatgpt.com/backend-api",
        )
    else:
        if model == "auto":
            from openai import OpenAI

            client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=60)
            ms = client.models.list().data
            if not ms:
                print("ERROR: no models", file=sys.stderr)
                return 2
            model = ms[0].id

        extra_body = None
        if args.reasoning_effort:
            extra_body = {"reasoning": {"effort": args.reasoning_effort}}
        judge = make_openai_judge(
            base_url=args.base_url,
            model=model,
            api_key=args.api_key,
            think_off=args.think_off,
            no_think=args.no_think,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            extra_body=extra_body,
        )
    pipeline = ScoutPipeline(args.repo)

    print(
        f"[tiered] n={len(queries)} model={model} top_n={args.top_n} "
        f"max_peeks={args.max_peeks} graphify={args.graphify} resume={len(results)}"
    )

    t0_all = time.time()
    for i in range(len(results), len(queries)):
        q = queries[i]
        query = q["query"]
        gold_files = _gold_files_for(q)
        gq = rebuild_query_gold(args.repo, q, max_lines=40)
        gt_code = gq.get("code") or ""
        gold_start = int(gq.get("code_start_line") or gq.get("start_line") or 0) or None
        gold_end = int(gq.get("code_end_line") or gold_start or 0) or None
        gold_range = (gold_start, gold_end) if gold_start and gold_end else None

        t0 = time.time()
        cap = run_tiered_scout(
            args.repo,
            query,
            judge,
            top_n=args.top_n,
            max_peeks=args.max_peeks,
            use_graphify=args.graphify,
            pipeline=pipeline,
        )
        elapsed = time.time() - t0

        cands = cap.candidates
        rank = _rank_in_candidates(gold_files, cands, stem_pair=False)
        file_ok = _file_hit(cap.file, gold_files, stem_pair=False) if cap.file else False
        scored = score_capsule(
            file_ok=file_ok,
            pred_code=cap.code or "",
            pred_lines=cap.lines,
            pred_symbol=cap.symbol,
            gold_code=gt_code,
            gold_range=gold_range,
            gold_start=gold_start,
            gold_symbol=q.get("symbol"),
        )

        row = {
            "query": query[:120],
            "gt_file": q.get("file"),
            "gold_files": gold_files,
            "status": cap.status,
            "file_ref": cap.file,
            "file_ok": file_ok,
            "lines": cap.lines,
            "symbol": cap.symbol,
            "code": (cap.code or "")[:4000],
            "gt_code": (gt_code or "")[:4000],
            "gold_start_line": gold_start,
            "gold_end_line": gold_end,
            "bleu": scored["bleu"],
            "line_iou": scored["line_iou"],
            "gold_start_hit": scored["gold_start_hit"],
            "symbol_hit": scored["symbol_hit"],
            "code_ok": scored["code_ok"],
            "code_ok_strict_bleu": scored["code_ok_strict_bleu"],
            "peeks": cap.peeks,
            "tried": cap.tried,
            "candidates": cands,
            "retrieval_rank": rank,
            "retrieval_hit_at_1": rank is not None and rank <= 1,
            "retrieval_hit_at_3": rank is not None and rank <= 3,
            "retrieval_hit_at_5": rank is not None and rank <= 5,
            "elapsed": round(elapsed, 2),
            "reason": cap.reason,
            "raw_judge": (cap.raw_judge or "")[:400],
        }
        results.append(row)

        n = len(results)
        f1 = sum(1 for r in results if r["file_ok"]) / n
        c1 = sum(1 for r in results if r["code_ok"]) / n
        c_strict = sum(1 for r in results if r.get("code_ok_strict_bleu")) / n
        r3 = sum(1 for r in results if r["retrieval_hit_at_3"]) / n
        ab = sum(r["bleu"] for r in results) / n
        aiou = sum(float(r.get("line_iou") or 0) for r in results) / n
        ash = sum(1 for r in results if r.get("gold_start_hit")) / n
        asym = sum(1 for r in results if r.get("symbol_hit")) / n
        print(
            f"  [{n}/{len(queries)}] file@1={f1:.1%} code={c1:.1%} bleu@0.8={c_strict:.1%} "
            f"start_hit={ash:.1%} iou={aiou:.3f} bleu={ab:.3f} peeks={cap.peeks} "
            f"status={cap.status} ({elapsed:.0f}s)"
        )

        summary = {
            "benchmark": "tiered_scout",
            "model": model,
            "n_samples": n,
            "n_target": len(queries),
            "top_n": args.top_n,
            "max_peeks": args.max_peeks,
            "file_match_rate": round(f1, 4),
            "code_match_rate": round(c1, 4),
            "code_match_rate_strict_bleu": round(c_strict, 4),
            "avg_bleu": round(ab, 4),
            "avg_line_iou": round(aiou, 4),
            "gold_start_hit_rate": round(ash, 4),
            "symbol_hit_rate": round(asym, 4),
            "file_at_3": round(r3, 4),
            "file_at_5": round(sum(1 for r in results if r["retrieval_hit_at_5"]) / n, 4),
            "avg_peeks": round(sum(r["peeks"] for r in results) / n, 2),
            "accept_rate": round(sum(1 for r in results if r["status"] == "accepted") / n, 4),
            "avg_scout_time_s": round(sum(r["elapsed"] for r in results) / n, 1),
            "wall_time_sec": round(time.time() - t0_all, 1),
            "base_url": args.base_url,
            "status": "partial" if n < len(queries) else "complete",
            "graphify": args.graphify,
            "scoring": "tight_gold+bleu+line_iou+start_hit",
        }
        _write_output(out_path, summary, results)
        if args.sleep > 0 and n < len(queries):
            time.sleep(args.sleep)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
