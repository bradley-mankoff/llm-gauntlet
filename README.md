# LLM Gauntlet — Local LLM Benchmarks

A hands-free benchmarking harness for locally-served and cloud models against
open-source instruction-following and code benchmarks.
Results below from a MacBook Pro M1 Max with 64 GB unified memory.

> **Not a formal submission.** These are samples (n=50–100 per benchmark), not
> exhaustive sweeps. The goal is practical, reproducible comparison of what's
> possible on a single Apple Silicon machine.

**Inference engines**: llama.cpp 9840 (Metal, kv-unified q8_0, 262k ctx, np=1), MTPLX 2.0.2 (MLX, native MTP depth 3), and cloud APIs.

---

## Role 1: Executor — Code Generation & Instruction Following

Can the model generate correct code from a prompt? Follow complex formatting instructions?

### Executor Results

| model | engine | IFEval strict | s/sample | HumanEval pass@1 | s/sample | cost/1M tok (in/out) |
|---|---|---|---|---|---|---|
| **DeepSeek v4 Flash** | cloud | **90%** | ~12s | **86%** | ~7s | $0.14 / $0.28 |
| Qwen3.6-27B MLX 4-bit | MTPLX | 85% | 23s | **88%** | 24s | $0 |
| Qwen3.6-27B Q4_K_XL | llama.cpp | 79% | 37s | **88%** | 28s | $0 |
| ThinkingCap Q4_K_M | llama.cpp | **86%** | 209s | 74% | 145s | $0 |
| Qwythos-9B MLX 4-bit | MTPLX | — | — | 78% | 26s | $0 |
| Ornith-35B-Q6_K | llama.cpp | 73% | 102s | 46% | 117s | $0 |
| **MiniCPM5-1B think-on** | llama.cpp | **63%** | 7s | **58%** | 6s | $0 |
| Qwen3.6-27B Q6_K_XL | llama.cpp | 80% | 40s | **88%** | 25s | $0 |
| Qwen3.5-122B-A10B Q2_K_XL | llama.cpp | 83% | 38s | 84% | 49s | $0 |

*DeepSeek Flash: thinking mode on both benchmarks. Published scores (full n=164): Flash Base 69.5%, Pro Base 76.8% HumanEval. No published IFEval for either. Our n=50 is noisier than full set. MiniCPM5 is a 1B Claude Opus fine-tune — 63% IFEval / 58% HumanEval is extraordinary for its size. Pricing accurate as of July 10, 2026.*

### Executor Key Findings

1. **DeepSeek v4 Flash is the IFEval leader** — 90% strict beats all local models. At $0.14/$0.28 per 1M tokens it's cheap enough for production, though local models remain free.
2. **Local 27B ties on HumanEval** — 88% matches the cloud model. Zero cost, on-premises.
3. **MiniCPM5 1B punches far above its weight** — 63% IFEval and 58% HumanEval from a 1B model at 6 s/sample. Better than Qwythos 9B think-on (45%). The Claude Opus + Fable 5 fine-tune works.
4. **Q4→Q6 buys nothing** — accuracy within noise for n=100/50.
5. **Both MTPLX and llama.cpp use MTP** — llama.cpp passes `--spec-type draft-mtp --spec-draft-n-max 2`. MTPLX uses depth 3 native MTP. Neither has a speed advantage; MTPLX wins on setup cleanliness.

---

## Role 2: Scout — Code Retrieval from Private Codebases

Given a codebase and a natural-language query, can the model find the right file and function? Full pipeline: embedder → reranker → scout LLM.

### Pipeline

**Embedder**: Qwen3-Embedding-0.6B (MTEB #1 for 0.6B class, instruction-aware)  
**Reranker**: BGE-Reranker-v2-m3 (cross-encoder; Qwen3-Reranker-4B hung on CPU)  
**Codebase**: MTPLX 2.0.2 source (149 Python files, 11 MB)  
**Queries**: 30 auto-generated from function docstrings across 30 files  
**Grading**: exact file name match (primary), BLEU on extracted code (secondary)

### Scout Results

| scout model | engine | file match | avg time | cost/30q |
|---|---|---|---|---|
| **Qwen3.6-27B MLX 4-bit** | MTPLX | **90.0%** | 138s | $0 |
| Qwythos-9B MLX 4-bit | MTPLX | 86.7% | 40s | $0 |
| DeepSeek v4 Pro (low reasoning) | cloud | 73.3% | 26s | ~$0.30 |
| DeepSeek v4 Pro (xhigh) | cloud | 33.3% | 65s | ~$0.21 |
| DeepSeek v4 Flash (xhigh) | cloud | 13.3% | 18s | ~$0.05 |
| MiniCPM5-1B (any mode) | llama.cpp | ~10% | — | $0 |

*Cloud costs estimated at DeepSeek API rates (July 2026): Pro $0.435/$0.87 per 1M tok in/out; Flash $0.14/$0.28. xhigh thinking on Pro/Flash caused models to consume all output tokens with internal reasoning — format compliance dropped to near zero. MiniCPM5 cannot handle large codebase-scale context.*

### Scout Key Findings

1. **Local models crush cloud on private codebase retrieval** — the 27B (90%) and Qwythos 9B (87%) both beat DeepSeek v4 Pro (73%). Cloud models hallucinate paths or ignore provided context.
2. **Qwythos 9B is the speed/accuracy sweet spot** — 86.7% file match at 40s/query, 3.5× faster than the 27B with only 3.3 points less accuracy.
3. **Thinking hurts scout accuracy** — adds latency with no accuracy gain. Causes DeepSeek to consume all output tokens with internal reasoning.
4. **Prompt engineering is critical for cloud models** — DeepSeek went from 33% (verbose format + xhigh) to 73% (simple "which file?" + low reasoning).
5. **Small models beat giants on narrow tasks** — Qwythos 9B outperforms DeepSeek v4 Pro at zero cost. Fine-tuned SLMs dominate domain-specific context-adherence (NVIDIA 2025, Forbes July 2026).
6. **The embedder matters less than the scout LLM** — the Qwen3-0.6B + BGE-v2-m3 stack is SOTA. The bottleneck is the scout LLM's ability to read provided context and follow instructions.

---

## Models Tested

| preset | model | params | quant | engine | size | role |
|---|---|---|---|---|---|---|
| qwen4 | Qwen3.6-27B-MTP | 27B | Q4_K_XL | llama.cpp | 16 GB | executor |
| mtplx | Qwen3.6-27B-MTPLX-Optimized-Speed | 27B | MLX 4-bit | MTPLX | 15 GB | executor + scout |
| thinkingcap | ThinkingCap-Qwen3.6-27B | 27B | Q4_K_M | llama.cpp | 16 GB | executor |
| qwen122b | Qwen3.5-122B-A10B-MTP (MoE) | 122B/10B active | UD-Q2_K_XL | llama.cpp | 43 GB | executor |
| ornith | Ornith-1.0-35B-Q6_K-Frankenstein-MTP | 35B | Q6_K | llama.cpp | 30 GB | executor |
| qwythos | Qwythos-9B-Claude-Mythos-5-MTPLX | 9B | MLX 4-bit | MTPLX | 5 GB | executor + scout |
| minicpm5 | MiniCPM5-1B-Claude-Opus-Fable5-Thinking | 1B | Q8_0 | llama.cpp | 1.1 GB | executor |
| ds-flash | DeepSeek v4 Flash | MoE | — | cloud API | — | executor + scout |
| ds-pro | DeepSeek v4 Pro | 1.6T/49B active | — | cloud API | — | scout |

---

## Benchmarks

### Executor Benchmarks

All three use seed=42, temperature=0.

| Benchmark | Dataset | n | Measures |
|---|---|---|---|
| IFEval | google/IFEval (541 prompts) | 100 | Instruction following (25 types, strict+loose) |
| HumanEval | openai/openai_humaneval (164 problems) | 50 | Code generation (execution-based, pass@1) |
| RepoQA | evalplus/repoqa_release (500 tasks) | 44-50 | Long-context code retrieval (BLEU ≥ 0.8) |

## Stress Test: Concurrent Executor + Scout

Running IFEval (27B executor) and scout pipeline (Qwythos 9B) simultaneously on the same M1 Max 64GB:

| Role | Solo s/sample | Concurrent s/sample | Slowdown |
|---|---|---|---|
| Executor (IFEval on 27B) | 22.8s | ~90s | **3.9×** |
| Scout (pipeline on Qwythos 9B) | 40s | ~56s | 1.4× |

**Memory: 63 GB / 64 GB used — 1 GB from OOM.** Running both models simultaneously is not viable on a single M1 Max. The 27B (15 GB) + Qwythos (5 GB) + embedder (1.2 GB) + reranker (2 GB) + context allocations saturate the machine. Recommendation: serve executor from the M1 Max, run scout (with a lighter model) on the main computer.
---

## Quick Start

```bash
git clone https://github.com/bradley-mankoff/llm-gauntlet.git
cd llm-gauntlet && uv sync

# Local executor (llama.cpp)
BENCH=1 MODEL_PRESET=qwen4 USE_FROGGERIC_CHAT_TEMPLATE=1 ./scripts/serve.sh
uv run python bench_one.py --benchmark humaneval --model auto --out results/h.json --n-samples 50 --max-tokens 2048 --seed 42

# Local executor (MTPLX)
mtplx serve --model Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed --port 8000 --reasoning off --no-stats-footer --profile turbo
uv run python bench_one.py --benchmark ifeval --model auto --base-url http://localhost:8000/v1 --out results/i.json --n-samples 100 --max-tokens 8192

# Scout pipeline (local)
uv run python bench_pipeline.py --model auto --base-url http://localhost:8000/v1 --repo /path/to/codebase --queries queries/scout_mtplx.json --out results/s.json

# Cloud executor
uv run python bench_one.py --benchmark humaneval --model deepseek-v4-flash --base-url https://api.deepseek.com/v1 --out results/h.json --n-samples 50 --max-tokens 2048
```

---

## Pricing Reference (July 2026)

| Model | Cache-hit input | Cache-miss input | Output |
|---|---|---|---|
| DeepSeek v4 Flash | $0.0028/M | $0.14/M | $0.28/M |
| DeepSeek v4 Pro | $0.0036/M | $0.435/M | $0.87/M |

Source: [DeepSeek official pricing](https://api-docs.deepseek.com/quick_start/pricing). All local models: $0.

---

## Known Limitations

- All models use temperature=0 (greedy). Sampling variance not measured.
- HumanEval n=50 (vs published full n=164) — noisier but directly comparable across our models.
- MTPLX turbo profile crashes under sustained reasoning load on M1 Max. Use `--profile sustained` for thinking-on.
- Scout pipeline code extraction (BLEU) is secondary to file match. BLEU threshold (≥0.8) is strict.
- Qwen3-Embedding-4B and Qwen3-Reranker-4B are MTEB leaders but impractical on CPU (75 min to index vs 5 min for 0.6B).

## Citation / Sources

- **IFEval**: Zhou et al. (2023). `google/IFEval`. Verifier: `google-research/instruction_following_eval` (Apache 2.0, vendored).
- **HumanEval**: Chen et al. (2021). `openai/openai_humaneval`.
- **RepoQA**: Tian et al. (ICML 2024). `evalplus/repoqa_release`.
- **MTPLX**: Youssof Altoukhi (2026). `github.com/youssofal/MTPLX`.
- **Qwen3-Embedding**: Alibaba (2025). MTEB #1 embedding series.
- **BGE-Reranker**: BAAI (2024). `BAAI/bge-reranker-v2-m3`.
- **DeepSeek V4**: DeepSeek (April 2026). BenchLM, official pricing page.
- **Chat template**: froggeric/Qwen-Fixed-Chat-Templates v21.3 (Apache 2.0).
- **Models**: Ornith (DeepReinforce AI), Qwen3.6 (Alibaba/unsloth), ThinkingCap (BottleCap AI), Qwythos (Empero AI), MiniCPM5 (OpenBMB/GnLOLot).

## License

Apache 2.0. The vendored IFEval verifier (`ifeval_lib/`) and froggeric chat template are also Apache 2.0.
