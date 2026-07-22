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
from scout_pipeline import ScoutPipeline  # noqa: E402
from tiered_scout import make_openai_judge, run_tiered_scout  # noqa: E402


def _bleu(gt_code: str, hyp: str) -> float:
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        if not gt_code.strip() or not hyp.strip():
            return 0.0
        return float(
            sentence_bleu(
                [gt_code.split()],
                hyp.split(),
                smoothing_function=SmoothingFunction().method1,
            )
        )
    except Exception:
        return 1.0 if gt_code.strip() and gt_code.strip() == hyp.strip() else 0.0


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
    if model == "auto":
        from openai import OpenAI

        client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=60)
        ms = client.models.list().data
        if not ms:
            print("ERROR: no models", file=sys.stderr)
            return 2
        model = ms[0].id

    judge = make_openai_judge(
        base_url=args.base_url,
        model=model,
        api_key=args.api_key,
        think_off=args.think_off,
        no_think=args.no_think,
        max_tokens=800,
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
        gt_code = q.get("code") or ""
        # Prefer gold symbol body from source when code missing.
        if not gt_code and q.get("file") and q.get("start_line"):
            try:
                lines = (Path(args.repo) / q["file"]).read_text(encoding="utf-8", errors="replace").splitlines()
                sl = max(0, int(q["start_line"]) - 1)
                gt_code = "\n".join(lines[sl : sl + 80])
            except Exception:
                gt_code = ""

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
        bleu = _bleu(gt_code, cap.code) if file_ok else 0.0
        code_ok = bool(file_ok and bleu >= 0.8)

        row = {
            "query": query[:120],
            "gt_file": q.get("file"),
            "gold_files": gold_files,
            "status": cap.status,
            "file_ref": cap.file,
            "file_ok": file_ok,
            "lines": cap.lines,
            "symbol": cap.symbol,
            "code": (cap.code or "")[:2000],
            "bleu": round(float(bleu), 4),
            "code_ok": code_ok,
            "peeks": cap.peeks,
            "tried": cap.tried,
            "candidates": cands,
            "retrieval_rank": rank,
            "retrieval_hit_at_1": rank is not None and rank <= 1,
            "retrieval_hit_at_3": rank is not None and rank <= 3,
            "retrieval_hit_at_5": rank is not None and rank <= 5,
            "elapsed": round(elapsed, 2),
            "reason": cap.reason,
            "raw_judge": cap.raw_judge[:400],
        }
        results.append(row)

        n = len(results)
        f1 = sum(1 for r in results if r["file_ok"]) / n
        c1 = sum(1 for r in results if r["code_ok"]) / n
        r3 = sum(1 for r in results if r["retrieval_hit_at_3"]) / n
        ab = sum(r["bleu"] for r in results) / n
        print(
            f"  [{n}/{len(queries)}] file@1={f1:.1%} code@0.8={c1:.1%} ret@3={r3:.1%} "
            f"bleu={ab:.3f} peeks={cap.peeks} status={cap.status} ({elapsed:.0f}s)"
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
            "avg_bleu": round(ab, 4),
            "file_at_3": round(r3, 4),
            "file_at_5": round(sum(1 for r in results if r["retrieval_hit_at_5"]) / n, 4),
            "avg_peeks": round(sum(r["peeks"] for r in results) / n, 2),
            "accept_rate": round(sum(1 for r in results if r["status"] == "accepted") / n, 4),
            "avg_scout_time_s": round(sum(r["elapsed"] for r in results) / n, 1),
            "wall_time_sec": round(time.time() - t0_all, 1),
            "base_url": args.base_url,
            "status": "partial" if n < len(queries) else "complete",
            "graphify": args.graphify,
        }
        _write_output(out_path, summary, results)
        if args.sleep > 0 and n < len(queries):
            time.sleep(args.sleep)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
