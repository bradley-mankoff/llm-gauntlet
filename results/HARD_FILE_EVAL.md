# Hard file-only eval snapshot

## Retrieval gate (no LLM)

### MTPLX docstring n=30 (graphify on)
- file@1: 0.9333
- file@3: 0.9667
- file@5: 1.0

### Prism hard cleaned n=40 (graphify off)
- file@1: 0.95
- file@3: 0.95
- file@5: 0.95

## Fable file-only on cleaned prism

```json
{
  "benchmark": "scout_pipeline",
  "model": "/Users/bradley_mankoff/llama-runs/models/fable-fusion-mtp-q4km/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q4_K_M.gguf",
  "n_samples": 40,
  "n_target": 40,
  "top_n": 5,
  "file_match_rate": 0.95,
  "file_at_1": 0.95,
  "file_at_3": 0.95,
  "file_at_5": 0.95,
  "file_at_3_stem": 0.95,
  "code_match_rate": 0.0,
  "avg_bleu": 0.0,
  "avg_scout_time_s": 61.0,
  "wall_time_sec": 2029.2,
  "base_url": "http://127.0.0.1:8080/v1",
  "status": "complete",
  "file_only": true
}
```

Misses (2/40): gold not in top-5 candidates for both.


## Qwythos v2 vs Fable (cleaned prism hard, file-only, top_n=5)

| model | file@1 pick | ret@3 | ret@5 | s/q |
|---|---:|---:|---:|---:|
| Fable MTP Q4_K_M | 0.95 | 0.95 | 0.95 | 61.0 |
| **Qwythos v2 Q4_K_M** | **0.95** | **0.95** | **0.95** | **17.7** |

Retrieval ceiling on this set is 95% @5 (same index for both models).
Both models hit that ceiling on file@1 here — weaker/faster Qwythos matches Fable pick quality on this gate, ~3.4× faster.

Note: Qwythos needs llama-server `--reasoning off` (otherwise content is empty and FILE: parse fails).


## Tiered scout (retrieval top-5 → sequential judge peeks)

Pipeline: embedder+reranker(+graphify) → top-5 files → local Qwythos judge opens **one file window at a time** until ACCEPT with LINES+code.

| setup | file@1 | code@0.8 | avg BLEU | ret@5 | avg peeks | s/q |
|---|---:|---:|---:|---:|---:|---:|
| Fable single-shot full extract (graphify) | 90.0% | 50.0% | 0.601 | — | 1 | 141s |
| **Tiered Qwythos walk (graphify)** | **93.3%** | **73.3%** | **0.805** | **100%** | **1.0** | **22s** |

Artifact: `results/pipeline_tiered_qwythos_mtplx.json`.

OMP agent: `capsule-scout` (Composer) uses `scripts/local_scout_cli.py` for candidates/windows; orchestrator should `task` that agent with `{query, repo}`.


## Prism hard tiered Qwythos (rescored)

Retrieval top-5 + Qwythos sequential capsule. Gold code backfilled offline; capsules unchanged.

| metric | value |
|---|---:|
| file@1 | 92.5% |
| code@0.8 | 40.0% |
| code@0.5 | 47.5% |
| avg BLEU | 0.452 |
| code@0.8 \| file_ok | 43.2% |
| avg BLEU \| file_ok | 0.488 |
| gold start in pred range \| file_ok | 19/37 |
| avg peeks | 1.07 |
| s/q | 18.6s |

Artifact: `results/pipeline_tiered_qwythos_prism_hard_clean.json`.

## Prism tiered rescored (tight gold + region metrics)

| judge | file@1 | code_ok (region) | bleu@0.8 | start_hit | avg IoU | avg BLEU | sym_hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwythos | 92.5% | 67.5% | 45.0% | 65.0% | 0.536 | 0.554 | 85.0% |
| laguna | 95.0% | 67.5% | 35.0% | 52.5% | 0.523 | 0.508 | 90.0% |
| composer | 92.5% | 70.0% | 42.5% | 70.0% | 0.514 | 0.520 | 90.0% |
| luna_high | 92.5% | 67.5% | 40.0% | 65.0% | 0.499 | 0.497 | 87.5% |
| luna_max | 95.0% | 67.5% | 35.0% | 52.5% | 0.490 | 0.476 | 90.0% |

code_ok = file_ok and (BLEU>=0.8 or line_IoU>=0.5 or gold_start in pred range).
Gold blocks tightened to ~40-line definition slices. Capsules re-read from disk via LINES.

### Approach fix (code)
- Judge may return `MORE` + `CENTER_LINE` for within-file retarget (up to 3 windows/file).
- Auto anchors: retrieval start + symbol hits + query-keyword hits.
- Capsule code sliced from source by LINES (not model paste).

