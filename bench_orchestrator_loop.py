#!/usr/bin/env python3
"""Multi-round scout benchmark with code-only orchestrator judge."""
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
    _write_output,
)
from orchestrator_judge import (  # noqa: E402
    assert_no_gold_leak,
    code_judge,
    estimate_cost_usd,
)
from scout_metrics import rebuild_query_gold  # noqa: E402
from scout_pipeline import ScoutPipeline  # noqa: E402
from tiered_scout import (  # noqa: E402
    UsageMeter,
    make_codex_judge,
    make_openai_judge,
    run_tiered_scout,
)


def _dedupe_append(history: list[str], feedback: str) -> None:
    """Append feedback lines to history, skipping exact duplicates."""
    text = (feedback or "").strip()
    if not text:
        return
    seen = set(history)
    for line in text.splitlines():
        line = line.strip()
        if line and line not in seen:
            history.append(line)
            seen.add(line)


def _meter_snap(meter: UsageMeter) -> tuple[int, int]:
    return int(getattr(meter, "prompt_tokens", 0) or 0), int(
        getattr(meter, "completion_tokens", 0) or 0
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Multi-round tiered-scout bench with code-only orchestrator judge"
    )
    ap.add_argument("--repo", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--max-peeks", type=int, default=5)
    ap.add_argument("--graphify", action="store_true")
    ap.add_argument(
        "--max-rounds",
        "-K",
        type=int,
        default=3,
        dest="max_rounds",
        help="Max scout rounds per query (success@K)",
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="auto")
    ap.add_argument("--api-key", default="sk-noop")
    ap.add_argument(
        "--judge-backend",
        default="openai",
        choices=["openai", "codex"],
        help="openai=chat completions; codex=ChatGPT Codex Responses SSE",
    )
    ap.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        help="OpenRouter/OpenAI-style reasoning.effort for thinking models",
    )
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--think-off", action="store_true")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument(
        "--price-in",
        type=float,
        default=0.0,
        help="USD per 1M prompt tokens (default 0)",
    )
    ap.add_argument(
        "--price-out",
        type=float,
        default=0.0,
        help="USD per 1M completion tokens (default 0)",
    )
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--resume", action="store_true")
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
    meter = UsageMeter()
    if args.judge_backend == "codex":
        if model == "auto":
            model = "gpt-5.6-luna"
        effort = args.reasoning_effort or "high"
        base = args.base_url
        if "chatgpt.com" not in base and not base.endswith("/codex"):
            base = "https://chatgpt.com/backend-api"
        judge = make_codex_judge(
            model=model,
            api_key=args.api_key,
            reasoning_effort=effort,
            timeout=args.timeout,
            base_url=base,
            meter=meter,
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
            meter=meter,
        )

    pipeline = ScoutPipeline(args.repo)
    k = max(1, int(args.max_rounds))

    print(
        f"[orch-loop] n={len(queries)} model={model} K={k} top_n={args.top_n} "
        f"max_peeks={args.max_peeks} graphify={args.graphify} resume={len(results)}"
    )

    t0_all = time.time()
    summary: dict = {}

    for i in range(len(results), len(queries)):
        q = queries[i]
        query = q["query"]
        gold_files = _gold_files_for(q)
        gq = rebuild_query_gold(args.repo, q, max_lines=40)
        gt_code = gq.get("code") or ""
        gold_start = int(gq.get("code_start_line") or gq.get("start_line") or 0) or None
        gold_end = int(gq.get("code_end_line") or gold_start or 0) or None
        gold_range = (gold_start, gold_end) if gold_start and gold_end else None
        gold_symbol = q.get("symbol")

        snap_p, snap_c = _meter_snap(meter)
        t0 = time.time()
        feedback_hist: list[str] = []
        rounds_detail: list[dict] = []
        success = False
        first_shot_code_ok = False
        final_scores = {
            "bleu": 0.0,
            "line_iou": 0.0,
            "gold_start_hit": False,
            "code_ok": False,
        }
        final_cap = None
        final_jr = None
        final_file_ok = False

        for rnd in range(1, k + 1):
            feedback_str = "\n".join(feedback_hist)
            try:
                cap = run_tiered_scout(
                    args.repo,
                    query,
                    judge,
                    top_n=args.top_n,
                    max_peeks=args.max_peeks,
                    use_graphify=args.graphify,
                    pipeline=pipeline,
                    feedback=feedback_str or None,
                )
            except TypeError:
                # Fallback if feedback kw not yet wired
                cap = run_tiered_scout(
                    args.repo,
                    query,
                    judge,
                    top_n=args.top_n,
                    max_peeks=args.max_peeks,
                    use_graphify=args.graphify,
                    pipeline=pipeline,
                )

            file_ok = _file_hit(cap.file, gold_files, stem_pair=False) if cap.file else False
            jr = code_judge(
                file_ok=file_ok,
                status=cap.status,
                pred_file=cap.file,
                pred_lines=cap.lines,
                pred_code=cap.code or "",
                pred_symbol=cap.symbol,
                gold_files=gold_files,
                gold_code=gt_code,
                gold_range=gold_range,
                gold_start=gold_start,
                gold_symbol=gold_symbol,
            )
            scores = jr.scores or {}
            if rnd == 1:
                first_shot_code_ok = bool(jr.accept or scores.get("code_ok"))

            rounds_detail.append(
                {
                    "round": rnd,
                    "file_ok": file_ok,
                    "accept": bool(jr.accept),
                    "tags": list(jr.tags or []),
                    "bleu": scores.get("bleu", 0.0),
                    "line_iou": scores.get("line_iou", 0.0),
                    "gold_start_hit": bool(scores.get("gold_start_hit")),
                    "peeks": cap.peeks,
                    "feedback": jr.feedback or "",
                    "status": cap.status,
                    "file_ref": cap.file,
                    "lines": cap.lines,
                    "symbol": cap.symbol,
                }
            )

            final_cap = cap
            final_jr = jr
            final_file_ok = file_ok
            final_scores = {
                "bleu": scores.get("bleu", 0.0),
                "line_iou": scores.get("line_iou", 0.0),
                "gold_start_hit": bool(scores.get("gold_start_hit")),
                "code_ok": bool(scores.get("code_ok") or jr.accept),
            }

            if jr.accept:
                success = True
                break

            fb = jr.feedback or ""
            assert_no_gold_leak(fb, gold_files, gold_symbol, gold_range)
            _dedupe_append(feedback_hist, fb)

        elapsed = time.time() - t0
        end_p, end_c = _meter_snap(meter)
        prompt_tok = max(0, end_p - snap_p)
        completion_tok = max(0, end_c - snap_c)
        dollars = float(
            estimate_cost_usd(
                prompt_tok,
                completion_tok,
                args.price_in,
                args.price_out,
            )
        )

        cap = final_cap
        jr = final_jr
        row = {
            "query": query[:120],
            "gt_file": q.get("file"),
            "gold_files": gold_files,
            "success": success,
            "rounds": len(rounds_detail),
            "dollars": round(dollars, 6),
            "seconds": round(elapsed, 2),
            "prompt_tokens": prompt_tok,
            "completion_tokens": completion_tok,
            "first_shot_code_ok": first_shot_code_ok,
            "rounds_detail": rounds_detail,
            "status": (cap.status if cap else "error"),
            "file_ref": (cap.file if cap else None),
            "file_ok": final_file_ok,
            "lines": (cap.lines if cap else None),
            "symbol": (cap.symbol if cap else None),
            "code": ((cap.code or "")[:4000] if cap else ""),
            "gt_code": (gt_code or "")[:4000],
            "gold_start_line": gold_start,
            "gold_end_line": gold_end,
            "bleu": final_scores["bleu"],
            "line_iou": final_scores["line_iou"],
            "gold_start_hit": final_scores["gold_start_hit"],
            "code_ok": final_scores["code_ok"],
            "peeks": (cap.peeks if cap else 0),
            "tried": (cap.tried if cap else []),
            "candidates": (cap.candidates if cap else []),
            "reason": (cap.reason if cap else ""),
            "accept_tags": list(jr.tags or []) if jr else [],
            "final_feedback": (jr.feedback if jr else "") or "",
        }
        results.append(row)

        n = len(results)
        n_ok = sum(1 for r in results if r["success"])
        success_at_k = n_ok / n
        avg_dollars = sum(float(r["dollars"]) for r in results) / n
        avg_seconds = sum(float(r["seconds"]) for r in results) / n
        avg_rounds_all = sum(int(r["rounds"]) for r in results) / n
        succ_rounds = [int(r["rounds"]) for r in results if r["success"]]
        avg_rounds_to_success = (
            sum(succ_rounds) / len(succ_rounds) if succ_rounds else None
        )
        first_shot = sum(1 for r in results if r.get("first_shot_code_ok")) / n

        print(
            f"  [{n}/{len(queries)}] success@K={success_at_k:.1%} "
            f"$/task={avg_dollars:.4f} s={row['seconds']:.0f} rounds={row['rounds']}"
        )

        summary = {
            "benchmark": "orchestrator_loop",
            "model": model,
            "n_samples": n,
            "n_target": len(queries),
            "max_rounds": k,
            "success_at_k": round(success_at_k, 4),
            "avg_dollars": round(avg_dollars, 6),
            "total_dollars": round(sum(float(r["dollars"]) for r in results), 6),
            "avg_seconds": round(avg_seconds, 2),
            "total_seconds": round(sum(float(r["seconds"]) for r in results), 2),
            "avg_rounds_to_success": (
                round(avg_rounds_to_success, 3)
                if avg_rounds_to_success is not None
                else None
            ),
            "avg_rounds_all": round(avg_rounds_all, 3),
            "first_shot_success_rate": round(first_shot, 4),
            "status": "partial" if n < len(queries) else "complete",
            "price_in": args.price_in,
            "price_out": args.price_out,
            "top_n": args.top_n,
            "max_peeks": args.max_peeks,
            "graphify": args.graphify,
            "base_url": args.base_url,
            "judge_backend": args.judge_backend,
            "wall_time_sec": round(time.time() - t0_all, 1),
            "scoring": (
                "code_judge accept iff file_ok and "
                "(bleu>=0.8 or line_iou>=0.5 or gold_start_hit); "
                "leak-safe template feedback; dollars=tokens*price"
            ),
        }
        _write_output(out_path, summary, results)
        if args.sleep > 0 and n < len(queries):
            time.sleep(args.sleep)

    if not summary and results:
        # resume already complete
        n = len(results)
        n_ok = sum(1 for r in results if r.get("success"))
        succ_rounds = [int(r["rounds"]) for r in results if r.get("success")]
        summary = {
            "benchmark": "orchestrator_loop",
            "model": model,
            "n_samples": n,
            "n_target": len(queries),
            "max_rounds": k,
            "success_at_k": round(n_ok / n, 4) if n else 0.0,
            "avg_dollars": round(sum(float(r.get("dollars") or 0) for r in results) / n, 6)
            if n
            else 0.0,
            "total_dollars": round(sum(float(r.get("dollars") or 0) for r in results), 6),
            "avg_seconds": round(sum(float(r.get("seconds") or 0) for r in results) / n, 2)
            if n
            else 0.0,
            "total_seconds": round(sum(float(r.get("seconds") or 0) for r in results), 2),
            "avg_rounds_to_success": (
                round(sum(succ_rounds) / len(succ_rounds), 3) if succ_rounds else None
            ),
            "avg_rounds_all": round(sum(int(r.get("rounds") or 0) for r in results) / n, 3)
            if n
            else 0.0,
            "first_shot_success_rate": round(
                sum(1 for r in results if r.get("first_shot_code_ok")) / n, 4
            )
            if n
            else 0.0,
            "status": "complete" if n >= len(queries) else "partial",
            "price_in": args.price_in,
            "price_out": args.price_out,
            "scoring": (
                "code_judge accept iff file_ok and "
                "(bleu>=0.8 or line_iou>=0.5 or gold_start_hit); "
                "leak-safe template feedback; dollars=tokens*price"
            ),
        }

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
