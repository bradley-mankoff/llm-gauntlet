"""Scout pipeline — MiniCPM5 1B via llama.cpp"""
import json, time, sys
sys.path.insert(0, '.')
from openai import OpenAI
from scout_pipeline import ScoutPipeline, _parse_file_ref
from pathlib import Path

REPO = "/opt/homebrew/var/mtplx/venv-2.0.2/lib/python3.13/site-packages/mtplx"
MODEL = "GnLOLot/MiniCPM5-1B-Claude-Opus-Fable5-Thinking-GGUF"

client = OpenAI(api_key="sk-noop", base_url="http://localhost:8080/v1")
pipeline = ScoutPipeline(REPO)

with open("queries/scout_mtplx.json") as f:
    queries = json.load(f)

results = []
file_matches = 0

for i, q in enumerate(queries):
    chunks = pipeline.retrieve(q["query"], top_k=10, rerank_top_n=5)
    prompt = pipeline.build_scout_prompt(q["query"], chunks)

    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Do NOT think. Answer directly. Output only the requested format."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=2048, temperature=0.0,
        extra_body={"enable_thinking": False},
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
        "tokens": resp.usage.completion_tokens,
    })

    if (i + 1) % 5 == 0:
        print(f"  [{i+1}/{len(queries)}] file={file_matches/(i+1):.1%} ({elapsed:.0f}s)")

wall = round(time.time() - t0, 1)
summary = {
    "benchmark": "scout_pipeline_minicpm5_1b",
    "model": MODEL,
    "n_samples": len(queries),
    "file_match_rate": round(file_matches / len(queries), 4),
    "avg_scout_time_s": round(sum(r["elapsed"] for r in results) / len(queries), 1),
    "wall_time_sec": wall,
}

with open("results/scout_minicpm5_1b.json", "w") as f:
    json.dump({"summary": summary, "results": results}, f, indent=2)

print(f"\nDONE: file_match={summary['file_match_rate']:.1%} avg={summary['avg_scout_time_s']:.0f}s wall={wall}s")
