"""Run scout pipeline against DeepSeek v4 Pro — simple format, low reasoning."""
import json, time, sys, re
sys.path.insert(0, '.')
from openai import OpenAI
from scout_pipeline import ScoutPipeline
from pathlib import Path

API_KEY = "sk-29114a6f095f42449e2732b341029b81"
REPO = "/opt/homebrew/var/mtplx/venv-2.0.2/lib/python3.13/site-packages/mtplx"

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com/v1")
pipeline = ScoutPipeline(REPO)

with open("queries/scout_mtplx.json") as f:
    queries = json.load(f)

def parse_file_ref(text):
    for pattern in [
        r'FILE:\s*(\S+)',
        r'`([\w/\-_.]+\.\w+)`',
        r'([\w/\-_.]+\.py)',
    ]:
        m = re.search(pattern, text)
        if m:
            ref = m.group(1).strip()
            if '.' in ref:
                return ref
    return None

results = []
file_matches = 0
total_cost = 0.0
PRICE_IN = 0.55 / 1_000_000
PRICE_OUT = 2.19 / 1_000_000
PRICE_CACHE = 0.14 / 1_000_000

for i, q in enumerate(queries):
    chunks = pipeline.retrieve(q["query"], top_k=10, rerank_top_n=5)

    parts = []
    for c in chunks:
        try:
            content = open(f"{REPO}/{c['file']}").read()[:15000]
        except:
            content = "[err]"
        parts.append(f"// {c['file']}\n{content}")

    prompt = f"Codebase:\n\n{chr(10).join(parts)}\n\nQuery: {q['query']}\n\nWhich file contains the answer? Reply with just the file path."

    t0 = time.time()
    resp = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512, temperature=0.0, reasoning_effort="low",
    )
    elapsed = time.time() - t0
    content = resp.choices[0].message.content or ""
    usage = resp.usage
    rt = usage.completion_tokens_details.reasoning_tokens

    cost = (usage.prompt_cache_hit_tokens * PRICE_CACHE +
            usage.prompt_cache_miss_tokens * PRICE_IN +
            usage.completion_tokens * PRICE_OUT)
    total_cost += cost

    file_ref = parse_file_ref(content)
    file_ok = file_ref and Path(file_ref).name == Path(q["file"]).name
    if file_ok:
        file_matches += 1

    results.append({
        "query": q["query"][:80], "gt_file": q["file"],
        "file_ref": file_ref, "file_ok": file_ok,
        "elapsed": round(elapsed, 1), "cost": round(cost, 6),
        "reasoning_tokens": rt,
    })

    if (i + 1) % 5 == 0:
        print(f"  [{i+1}/{len(queries)}] file={file_matches/(i+1):.1%} cost=\${total_cost:.4f} ({elapsed:.0f}s)")

wall = round(time.time() - t0, 1)
summary = {
    "benchmark": "scout_pipeline_ds_pro_low",
    "model": "deepseek-v4-pro",
    "n_samples": len(queries),
    "file_match_rate": round(file_matches / len(queries), 4),
    "avg_scout_time_s": round(sum(r["elapsed"] for r in results) / len(queries), 1),
    "wall_time_sec": wall,
    "total_cost_usd": round(total_cost, 4),
}

with open("results/scout_ds_pro_low.json", "w") as f:
    json.dump({"summary": summary, "results": results}, f, indent=2)

print(f"\nDONE: file_match={summary['file_match_rate']:.1%} avg={summary['avg_scout_time_s']:.0f}s cost=\${summary['total_cost_usd']:.4f}")
