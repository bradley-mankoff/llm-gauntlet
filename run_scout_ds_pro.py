"""Run scout pipeline against DeepSeek v4 Pro — xhigh thinking, with cost tracking."""
import json, time, sys
sys.path.insert(0, '.')
from openai import OpenAI
from scout_pipeline import ScoutPipeline, _extract_code_block, _parse_file_ref
from pathlib import Path

API_KEY = "sk-29114a6f095f42449e2732b341029b81"
REPO = "/opt/homebrew/var/mtplx/venv-2.0.2/lib/python3.13/site-packages/mtplx"

# DeepSeek pricing per 1M tokens (as of 2026-07)
# v4 Pro: $0.55/M input, $2.19/M output (cache hit: $0.14/M)
PRICE_INPUT = 0.55 / 1_000_000
PRICE_OUTPUT = 2.19 / 1_000_000
PRICE_CACHE_HIT = 0.14 / 1_000_000

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com/v1")
pipeline = ScoutPipeline(REPO)

with open("queries/scout_mtplx.json") as f:
    queries = json.load(f)

results = []
file_matches = 0
total_cost = 0.0

for i, q in enumerate(queries):
    chunks = pipeline.retrieve(q["query"], top_k=10, rerank_top_n=5)
    prompt = pipeline.build_scout_prompt(q["query"], chunks)

    t0 = time.time()
    resp = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.0,
        reasoning_effort="high",
    )
    elapsed = time.time() - t0
    content = resp.choices[0].message.content or ""

    # Cost
    usage = resp.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    cache_hit = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
    cache_miss = getattr(usage, "prompt_cache_miss_tokens", 0) or 0

    cost = (cache_hit * PRICE_CACHE_HIT +
            cache_miss * PRICE_INPUT +
            completion_tokens * PRICE_OUTPUT)
    total_cost += cost

    file_ref = _parse_file_ref(content)
    file_ok = file_ref and Path(file_ref).name == Path(q["file"]).name
    if file_ok:
        file_matches += 1

    results.append({
        "query": q["query"][:80], "gt_file": q["file"],
        "file_ref": file_ref, "file_ok": file_ok,
        "elapsed": round(elapsed, 1),
        "cost": round(cost, 6),
        "tokens_in": prompt_tokens,
        "tokens_out": completion_tokens,
    })

    if (i + 1) % 5 == 0:
        print(f"  [{i+1}/{len(queries)}] file={file_matches/(i+1):.1%} cost=\${total_cost:.4f} ({elapsed:.0f}s)")

wall = round(time.time() - t0, 1)
summary = {
    "benchmark": "scout_pipeline_ds_pro_xhigh",
    "model": "deepseek-v4-pro",
    "n_samples": len(queries),
    "file_match_rate": round(file_matches / len(queries), 4),
    "avg_scout_time_s": round(sum(r["elapsed"] for r in results) / len(queries), 1),
    "wall_time_sec": wall,
    "total_cost_usd": round(total_cost, 4),
}

with open("results/scout_ds_pro_xhigh.json", "w") as f:
    json.dump({"summary": summary, "results": results}, f, indent=2)

print(f"\nDONE: file_match={summary['file_match_rate']:.1%} avg={summary['avg_scout_time_s']:.0f}s cost=\${summary['total_cost_usd']:.4f}")
