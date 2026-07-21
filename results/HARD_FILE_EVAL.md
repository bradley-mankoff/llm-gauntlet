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
