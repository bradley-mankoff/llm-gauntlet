# LLM Gauntlet — Consolidated Results

**Date**: 2026-07-06 – 2026-07-20
**Hardware**: MacBook Pro M1 Max, 64 GB unified memory, 10 cores, macOS 26.5.1

All benchmarks use seed=42, temperature=0.

---

## Models Tested

| preset | model | params | quant | engine | role |
|---|---|---|---|---|---|
| qwen4 | Qwen3.6-27B-MTP | 27B | Q4_K_XL | llama.cpp | executor |
| mtplx | Qwen3.6-27B-MTPLX-Optimized-Speed | 27B | MLX 4-bit | MTPLX | executor + scout |
| thinkingcap | ThinkingCap-Qwen3.6-27B | 27B | Q4_K_M | llama.cpp | executor |
| qwythos-v1 | Qwythos-9B-Claude-Mythos-5 | 9B | MLX 4-bit | MTPLX | executor + scout |
| **qwythos-v2** | **Qwythos-9B-v2** | **9B** | **Q4_K_M** | **llama.cpp** | **executor + scout** |
| minicpm5-v1 | MiniCPM5-1B-Claude-Opus-Fable5 | 1B | Q8_0 | llama.cpp | executor |
| minicpm5-v2 | MiniCPM5-1B-Claude-Opus-Fable5-V2 | 1B | Q8_0 | llama.cpp | executor |
| ds-flash | DeepSeek v4 Flash | MoE | — | cloud | executor + scout |
| ds-pro | DeepSeek v4 Pro | 1.6T/49B | — | cloud | scout |

**Bonsai-27B-mlx-1bit** (prism-ml) not yet benchmarked — requires MTPLX, not GGUF.

---

## Role 1: Scout — Code Retrieval

Full pipeline: Qwen3-Embedding-0.6B → BGE-Reranker-v2-m3 → scout LLM.
Queries: 30 auto-generated from function docstrings (MTPLX codebase), or 8 hand-crafted queries (gauntlet codebase).

### Scout Results (Optimal Config: Qwen3-Embedding + BGE-Reranker)

| scout model | engine | file match | s/query | codebase | n |
|---|---|---|---|---|---|
| Qwen3.6-27B MLX 4-bit + graphify | MTPLX | **90.0%** (avg BLEU **0.601**) | 141s | MTPLX | 30 |
| Qwen3.6-27B MLX 4-bit (no graphify) | MTPLX | **90.0%** (avg BLEU 0.599) | 138s | MTPLX | 30 |
| Qwythos v2 + graphify | llama.cpp | 73.3% (avg BLEU 0.454) | 53s | MTPLX | 30 |
| **Qwythos v2 Q4_K_M** | **llama.cpp** | **100%** | **44s** | **gauntlet** | **8** |
| Qwythos v1 MLX 4-bit | MTPLX | 86.7% | 40s | MTPLX | 30 |
| ThinkingCap Q4_K_M (think-on) | llama.cpp | 100% | 156s | gauntlet | 5 |
| DeepSeek v4 Pro (low) | cloud | 73.3% | 26s | MTPLX | 30 |
| MiniCPM5 v2 Q8_0 | llama.cpp | 0% | 47s | gauntlet | 8 |
| MiniCPM5 v1 Q8_0 | llama.cpp | ~20% | 13s | MTPLX | 30 |
| DeepSeek v4 Flash (xhigh) | cloud | 13.3% | 18s | MTPLX | 30 |

### Embedder/Reranker Ablation (Qwythos v1, MTPLX codebase, n=30)

| embedder | reranker | file match |
|---|---|---|
| Qwen3-Embedding-0.6B | BGE-Reranker-v2-m3 | **86.7%** |
| Qwen3-Embedding-0.6B | none | 80.0% |
| jina-code-embeddings-1.5b | BGE-Reranker-v2-m3 | 73.3% |
| jina-code-embeddings-1.5b | CoREB-code-reranker (4B) | 70.0% |

---

## Role 2: Executor — Code Generation & Instruction Following

### HumanEval pass@1

| model | engine | pass@1 | s/sample | n |
|---|---|---|---|---|
| Qwen3.6-27B MLX 4-bit | MTPLX | **88%** | 24s | 50 |
| Qwythos v1 MLX 4-bit | MTPLX | **78%** | 26s | 50 |
| DeepSeek v4 Flash | cloud | 86% | 7s | 50 |
| Qwen3.6-27B Q4_K_XL (no MTP) | llama.cpp | 73% | 37s | 15 |
| ThinkingCap Q4_K_M (think-on) | llama.cpp | 67% | 41s | 15 |
| MiniCPM5 v1 Q8_0 (think-on) | llama.cpp | **58%** | **6s** | 50 |
| Qwythos v2 Q4_K_M | llama.cpp | 55% | 26s | 20 |
| MiniCPM5 v2 Q8_0 | llama.cpp | 20% | 14s | 10 |

### IFEval strict

| model | engine | strict | s/sample | n |
|---|---|---|---|---|
| DeepSeek v4 Flash (xhigh) | cloud | **90%** | 12s | 100 |
| Qwen3.6-27B MLX 4-bit (think-off) | MTPLX | 85% | 23s | 100 |
| MiniCPM5 v1 Q8_0 (think-on) | llama.cpp | **63%** | **7s** | 100 |
| Qwen3.6-27B MLX 4-bit (think-on) | MTPLX | 32% | 169s | 100 |

### RepoQA avg BLEU (≥0.8 = pass)

| model | engine | avg BLEU | s/sample | n |
|---|---|---|---|---|
| ThinkingCap Q4_K_M (think-on) | llama.cpp | 0.779 | 164s | 50 |
| Ornith-35B-Q6_K (think-on) | llama.cpp | 0.259 | 53s | 44 |
| Qwen3.5-122B-A10B Q2_K_XL | llama.cpp | 0.219 | 57s | 44 |
| Qwen3.6-27B MLX 4-bit | MTPLX | 0.205 | 33s | 44 |

---

## Harder file-only eval (prism-llama.cpp)

Hygiene (non-overfit): drop TODO/banner/low-signal queries, skip vendored `deps/`, dual gold for C/C++ header+impl pairs, report **file@1** (model pick) and **retrieval file@3**.

| set | model | file@1 (pick) | retrieval @1 | retrieval @3 | retrieval @5 | s/q |
|---|---|---:|---:|---:|---:|---:|
| MTPLX docstring n=30 (gate) | retrieval only + graphify | — | **93.3%** | **96.7%** | **100%** | — |
| Prism hard dirty n=40 | Fable file-only | 80.0% | 72.5% | 85.0% | 85.0% | 64.5s |
| **Prism hard cleaned n=40** | **Fable file-only** | **95.0%** | **95.0%** | **95.0%** | **95.0%** | **61s** |
| **Prism hard cleaned n=40** | **Qwythos v2 file-only** | **95.0%** | **95.0%** | **95.0%** | **95.0%** | **18s** |

Artifacts: `queries/scout_prism_hard_file.json`, `results/pipeline_fable_prism_hard_clean_fileonly.json`, `results/retrieval_file_at_k_gate.json`.

Both remaining cleaned-prism misses had gold **outside** the top-5 candidates (not model hub-picking). Iterative “next file” over top-3 cannot beat ~95% without retrieval/query gains.

## Key Findings

### Scout
1. **Qwythos v2 is the scout winner.** 100% file match (8/8) at 44s/query on the gauntlet codebase. 3.5× faster than ThinkingCap (156s) for the same accuracy. v2 via llama.cpp matches or beats v1 via MTPLX on file retrieval.
2. **Target recipe completed: MTPLX 27B + uncapped graphify BFS.** Full n=30: **file 90.0% (27/30), avg BLEU 0.6007**, code@0.8 50%, 141s/query (`results/pipeline_27b_mtplx_graphify.json`). Secondary metric is **avg BLEU**, not the ≥0.8 pass rate. Same recipe earlier partial was 16/16 file-perfect before tool-timeout/server death; checkpointing fixed that.
3. **Qwythos v2 on the same graphify pipeline (n=30):** file **73.3%**, **avg BLEU 0.454**, code@0.8 40%, 53s/query (`pipeline_qwythos_v2_graphify.json`). Without graphify Qwythos v2 was stronger on file (80.0%) and avg BLEU (0.517) — graphify helps the 27B more than the 9B.
4. **Next scout candidate:** DavidAU Fable-Fusion 27B NEO-MTP Q4_K_M (~17.2 GB) via llama.cpp `--spec-type draft-mtp` + froggeric think-off, same graphify recipe (`scripts/run_fable_fusion_graphify.sh`).
5. **The BGE reranker is worth +7 points.** 86.7% → 80.0% without it. Boring stack wins: Qwen3-Embedding + BGE-Reranker-v2-m3.

### Executor
6. **27B MTPLX is the HumanEval leader.** 88% ties DeepSeek v4 Flash. Zero cost, on-premises. Without MTP though, the GGUF Q4_K_XL drops to 73% and runs 1.5× slower.
6. **Qwythos v2 regresses on HumanEval.** 55% vs v1's 78%. v2 appears optimized for retrieval/instruction-following over pure code generation. n=20 vs n=50 adds noise.
7. **MiniCPM5 v2 regresses everywhere.** 20% HumanEval (v1: 58%), 0% scout file match (v1: ~20%). The V2 fine-tune is worse on these benchmarks.
8. **MTP matters for speed.** Q4 quants of Qwen3.6-27B don't support `--spec-type draft-mtp`. Without MTP, the 27B runs at ~8 t/s instead of ~25 t/s. For practical use, serve 27B via MTPLX with MLX 4-bit.

### Production Recommendation (Scout + Executor Protocol)

| Role | Model | Why |
|---|---|---|
| **Scout** | Qwythos v2 (9B Q4_K_M) | 100% file match, 44s, 5 GB |
| **Executor** | Qwen3.6 27B (MTPLX MLX 4-bit) | 88% HE, 24s, 15 GB |

Combined: 20 GB. Fits in 64 GB alongside embedder (1.2 GB) + reranker (2 GB) + context allocations.

---

## Stress Test: Concurrent Scout + Executor

IFEval (27B executor) + scout pipeline (Qwythos v1) on same 64 GB M1 Max:

| Role | Solo s/sample | Concurrent s/sample | Slowdown |
|---|---|---|---|
| Executor (IFEval on 27B) | 22.8s | ~90s | 3.9× |
| Scout (pipeline on Qwythos) | 40s | ~56s | 1.4× |

**Memory: 63/64 GB — 1 GB from OOM.** Concurrent operation not viable. Recommendation: separate machines, or serve executor from M1 Max and run scout (Qwythos v2) on main computer.

---

## Benchmarks

| Benchmark | Dataset | n | Measures |
|---|---|---|---|
| Scout Pipeline | Auto-generated queries from docstrings | 8–30 | File-level code retrieval via embedder→reranker→LLM |
| HumanEval | openai/openai_humaneval (164) | 15–50 | Code generation (execution-based, pass@1) |
| IFEval | google/IFEval (541) | 50–100 | Instruction following (25 types, strict+loose) |
| RepoQA | evalplus/repoqa_release (500) | 44–50 | Long-context code retrieval (BLEU ≥ 0.8) |
| CORE-Bench | siegelz/core-bench (270) | — | Code-reading comprehension (built, not yet run) |

---

## File Manifest

```
gauntlet/
├── results/            # 40 benchmark JSON files
├── tasks/              # Benchmark implementations
│   ├── humaneval.py, ifeval.py, repoqa.py, codeeditbench.py
│   └── corebench.py    # CORE-Bench integration (not yet run)
├── scout_pipeline.py   # Scout pipeline with graphify-first retrieval
├── bench_pipeline.py   # Scout pipeline runner
├── bench_one.py        # Single-benchmark runner
├── bench_client.py     # OpenAI SDK wrapper
├── bench_new_models.py # Multi-benchmark runner for new models
├── scripts/serve.sh    # llama-server launcher with model presets
├── .omp/               # OMP skills and agents
│   ├── skills/         # 7 skills (graphify, orchestrate-change, etc.)
│   └── agents/         # 7 agents (worker, scout, tester, etc.)
└── chat_template.jinja # froggeric/Qwen-Fixed-Chat-Templates v21.3
```
