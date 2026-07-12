"""Run scout pipeline against DeepSeek v4 Flash."""
import json, time, sys
sys.path.insert(0, '.')
from openai import OpenAI
from scout_pipeline import ScoutPipeline, _extract_code_block, _parse_file_ref
from pathlib import Path

API_KEY = "sk-29114a6f095f42449e2732b341029b81"
REPO = "/opt/homebrew/var/mtplx/venv-2.0.2/lib/python3.13/site-packages/mtplx"
QUERIES = "queries/scout_mtplx.json"
OUT = "results/scout_ds_flash_pipeline.json"

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com/v1")
pipeline = ScoutPipeline(REPO)

with open(QUERIES) as f:
    queries = json.load(f)

results = []
file_matches = 0
t0_total = time.time()

for i, q in enumerate(queries):
    chunks = pipeline.retrieve(q["query"], top_k=10, rerank_top_n=5)
    prompt = pipeline.build_scout_prompt(q["query"], chunks)

    t0 = time.time()
    resp = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.0,
        reasoning_effort="high",
    )
    elapsed = time.time() - t0
    content = resp.choices[0].message.content or ""

    file_ref = _parse_file_ref(content)
    file_ok = file_ref and Path(file_ref).name == Path(q["file"]).name
    if file_ok:
        file_matches += 1

    results.append({
        "query": q["query"][:80], "gt_file": q["file"],
        "file_ref": file_ref, "file_ok": file_ok, "elapsed": round(elapsed, 1),
    })

    if (i + 1) % 5 == 0:
        print(f"  [{i+1}/{len(queries)}] file={file_matches/(i+1):.1%} ({elapsed:.0f}s)")

wall = round(time.time() - t0_total, 1)
summary = {
    "benchmark": "scout_pipeline_ds_flash_xhigh",
    "model": "deepseek-v4-flash",
    "n_samples": len(queries),
    "file_match_rate": round(file_matches / len(queries), 4),
    "avg_scout_time_s": round(sum(r["elapsed"] for r in results) / len(queries), 1),
    "wall_time_sec": wall,
}

with open(OUT, "w") as f:
    json.dump({"summary": summary, "results": results}, f, indent=2)

print(f"\nDONE: file_match={summary['file_match_rate']:.1%} avg={summary['avg_scout_time_s']:.0f}s wall={wall}s")
