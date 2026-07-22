#!/usr/bin/env python3
"""Rescore saved tiered scout results with tight-gold multi-metrics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scout_metrics import (  # noqa: E402
    rebuild_query_gold,
    score_capsule,
    slice_file_lines,
    parse_line_range,
)
from bench_pipeline import _file_hit, _gold_files_for  # noqa: E402


def rescore_one(path: Path, repo: Path, queries: list[dict], inplace: bool = True) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    by_q = {q["query"]: q for q in queries}
    # also allow truncated query keys from results
    by_prefix = {q["query"][:120]: q for q in queries}

    rows = []
    for r in d.get("results") or []:
        qtext = r.get("query") or ""
        q = by_q.get(qtext) or by_prefix.get(qtext[:120])
        if not q:
            # fuzzy: first query starting with
            q = next((qq for qq in queries if qq["query"].startswith(qtext[:80])), None)
        if not q:
            rows.append(r)
            continue

        gq = rebuild_query_gold(repo, q, max_lines=40)
        gold_files = _gold_files_for(q)
        gt_code = gq.get("code") or ""
        gold_start = int(gq.get("code_start_line") or 0) or None
        gold_end = int(gq.get("code_end_line") or gold_start or 0) or None
        gold_range = (gold_start, gold_end) if gold_start and gold_end else None

        file_ref = r.get("file_ref") or r.get("file")
        file_ok = _file_hit(file_ref, gold_files, stem_pair=False) if file_ref else False

        pred_code = r.get("code") or ""
        # Prefer disk slice from reported lines (handles truncated/missing excerpts)
        rng = parse_line_range(r.get("lines"))
        if file_ok and file_ref and rng:
            disk = slice_file_lines(repo / file_ref, rng[0], rng[1])
            if disk.strip():
                pred_code = disk

        scored = score_capsule(
            file_ok=file_ok,
            pred_code=pred_code,
            pred_lines=r.get("lines"),
            pred_symbol=r.get("symbol"),
            gold_code=gt_code,
            gold_range=gold_range,
            gold_start=gold_start,
            gold_symbol=q.get("symbol"),
        )
        r = dict(r)
        r["file_ok"] = file_ok
        r["gt_code"] = (gt_code or "")[:4000]
        r["code"] = (pred_code or "")[:4000]
        r["gold_start_line"] = gold_start
        r["gold_end_line"] = gold_end
        r.update(scored)
        rows.append(r)

    n = len(rows) or 1
    summary = dict(d.get("summary") or {})
    summary.update(
        {
            "n_samples": len(rows),
            "file_match_rate": round(sum(1 for r in rows if r.get("file_ok")) / n, 4),
            "code_match_rate": round(sum(1 for r in rows if r.get("code_ok")) / n, 4),
            "code_match_rate_strict_bleu": round(
                sum(1 for r in rows if r.get("code_ok_strict_bleu")) / n, 4
            ),
            "avg_bleu": round(sum(float(r.get("bleu") or 0) for r in rows) / n, 4),
            "avg_line_iou": round(sum(float(r.get("line_iou") or 0) for r in rows) / n, 4),
            "gold_start_hit_rate": round(sum(1 for r in rows if r.get("gold_start_hit")) / n, 4),
            "symbol_hit_rate": round(sum(1 for r in rows if r.get("symbol_hit")) / n, 4),
            "scoring": "tight_gold+bleu+line_iou+start_hit (offline rescore)",
        }
    )
    out = {"summary": summary, "results": rows}
    if inplace:
        path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--results", nargs="+", required=True)
    args = ap.parse_args()
    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    repo = Path(args.repo)
    for rp in args.results:
        p = Path(rp)
        if not p.exists():
            print("missing", p)
            continue
        s = rescore_one(p, repo, queries)
        print(
            p.name,
            "file", s.get("file_match_rate"),
            "code", s.get("code_match_rate"),
            "bleu@0.8", s.get("code_match_rate_strict_bleu"),
            "start_hit", s.get("gold_start_hit_rate"),
            "iou", s.get("avg_line_iou"),
            "bleu", s.get("avg_bleu"),
            "sym", s.get("symbol_hit_rate"),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
