"""End-to-end scout pipeline benchmark.

Runs the full embedder -> reranker -> scout LLM pipeline on a set of queries.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_client import list_models, make_client
from scout_pipeline import ScoutPipeline, _extract_code_block, _parse_file_ref


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="auto")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-n", type=int, default=3, help="Files to pass to scout LLM")
    args = ap.parse_args()

    client = make_client(base_url=args.base_url)
    models = list_models(client)
    if args.model == "auto":
        if len(models) != 1:
            print(f"ERROR: {len(models)} models, can't auto-pick", file=sys.stderr)
            return 2
        args.model = models[0]

    with open(args.queries) as f:
        queries = json.load(f)
    if args.n_samples:
        queries = queries[:args.n_samples]

    print(f"[pipeline] {len(queries)} queries, scout={args.model}, top_n={args.top_n}")

    pipeline = ScoutPipeline(args.repo)
    results = []
    file_matches = 0
    code_matches = 0
    total_time = 0.0
    total_bleu = 0.0

    t0_total = time.time()

    for i, q in enumerate(queries):
        # 1. Retrieve from embedder
        chunks = pipeline.retrieve(q["query"], top_k=10, rerank_top_n=args.top_n)
        prompt = pipeline.build_scout_prompt(q["query"], chunks)

        # 3. Scout LLM
        t0 = time.time()
        resp = _bench_chat(client, args.model, [{"role": "user", "content": prompt}],
                           max_tokens=args.max_tokens, temperature=0.0)
        elapsed = time.time() - t0
        total_time += elapsed

        response_text = resp.choices[0].message.content
        file_ref = _parse_file_ref(response_text)
        extracted = _extract_code_block(response_text)

        gt_file = q["file"]
        gt_code = q.get("code", "")
        file_ok = file_ref is not None and Path(file_ref).name == Path(gt_file).name

        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            bleu = sentence_bleu([gt_code.split()], extracted.split(),
                                 smoothing_function=SmoothingFunction().method1)
        except Exception:
            bleu = 1.0 if extracted.strip() == gt_code.strip() else 0.0

        code_ok = bleu >= 0.8
        if file_ok:
            file_matches += 1
        if code_ok:
            code_matches += 1
        total_bleu += bleu

        results.append({
            "query": q["query"][:100],
            "gt_file": gt_file,
            "file_ref": file_ref,
            "file_ok": file_ok,
            "bleu": round(bleu, 4),
            "code_ok": code_ok,
            "elapsed": round(elapsed, 1),
            "response": response_text[:300],
        })

        fm = file_matches / (i + 1)
        cm = code_matches / (i + 1)
        print(f"  [{i+1}/{len(queries)}] file={fm:.1%} code={cm:.1%} ({elapsed:.0f}s)")

    n = len(queries)
    wall = round(time.time() - t0_total, 1)
    summary = {
        "benchmark": "scout_pipeline",
        "model": args.model,
        "n_samples": n,
        "top_n": args.top_n,
        "file_match_rate": round(file_matches / n, 4) if n else 0,
        "code_match_rate": round(code_matches / n, 4) if n else 0,
        "avg_bleu": round(total_bleu / n, 4) if n else 0,
        "avg_scout_time_s": round(total_time / n, 1) if n else 0,
        "wall_time_sec": wall,
        "base_url": args.base_url,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\n[pipeline] done in {wall}s")
    print(json.dumps(summary, indent=2))
    return 0


# Late import to match module pattern
from bench_client import chat as _bench_chat

if __name__ == "__main__":
    sys.exit(main())
