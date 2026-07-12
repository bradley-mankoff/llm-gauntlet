# LLM Gauntlet — Local LLM Benchmarks

A hands-free benchmarking harness for locally-served models (llama.cpp GGUF and
MTPLX MLX) against open-source instruction-following and code benchmarks.
Results below from a MacBook Pro M1 Max with 64 GB unified memory.

> **Not a formal submission.** These are samples (n=50–100 per benchmark), not
> exhaustive sweeps. The goal is practical, reproducible comparison of what's
> possible on a single Apple Silicon machine.

## Results
**Inference engines**: llama.cpp 9840 (Metal, kv-unified q8_0, 262k ctx, np=1) and MTPLX 2.0.2 (MLX, native MTP depth 3, turbo profile)

| model | engine | IFEval strict | s/sample | HumanEval pass@1 | s/sample | RepoQA avg BLEU | s/sample | thinking |
|---|---|---|---|---|---|---|---|---|
| Ornith-35B-Q6_K | llama.cpp | 73% | 102 | 46% | 117 | 0.259 | 53 | on (bundled) |
| Qwen3.6-27B Q4_K_XL | llama.cpp | 79% | 37 | **88%** | 28 | 0.513 | — | off |
| Qwen3.6-27B Q6_K_XL | llama.cpp | 80% | 40 | **88%** | 25 | — | — | off |
| **Qwen3.6-27B MLX 4-bit** | **MTPLX** | **85%** | 23 | **88%** | 24 | 0.205 | 33 | off |
| Qwythos-9B MLX 4-bit | MTPLX | — | — | 78% | 26 | — | — | off |
| ThinkingCap Q4_K_M | llama.cpp | **86%** | 209 | 74% | 145 | **0.779** | 164 | on |
*Qwen4 RepoQA wall time missing. Qwen6 RepoQA data missing. MTPLX row uses turbo profile (sustained would be slower). Qwythos executor data is HumanEval only (thinking-off 78% at 26s/sample; thinking-on scored 45% and was cancelled). See Scout Pipeline section below for Qwythos scout results.*

## Key findings

### Executor benchmarks

1. **MTPLX dominates llama.cpp on IFEval (thinking off)** — 85% vs 79% strict, and 23 vs 37 s/sample. Same Qwen3.6-27B base, same MTP technique, but MLX 4-bit + native MTP depth 3 outruns GGUF Q4_K_XL + llama.cpp MTP depth 2.
2. **Both engines use MTP** — llama.cpp passes `--spec-type draft-mtp --spec-draft-n-max 2`. MTPLX uses depth 3 native MTP. Speeds are comparable; MTPLX's advantage is cleaner setup.
3. **Qwythos 9B is a scout, not an executor** — 78% HumanEval (think-off) is 10 points behind the 27B. Thinking mode drops it to 45%. But as a scout, it achieves 86.7% file match at 40s/query.
4. **Qwen4/Qwen6 are the speed winners** — 88% HumanEval at 25-28 s/sample. GPT-4 territory.
5. **Q4→Q6 buys nothing** — accuracy within noise for n=100/50.

### Scout pipeline

6. **Local models crush cloud on code retrieval** — 27B (90%) and Qwythos 9B (87%) beat DeepSeek v4 Pro (73%) on private codebase retrieval. Cloud models hallucinate paths or ignore provided context.
7. **Small models beat giants on narrow tasks** — Qwythos 9B outperforms DeepSeek v4 Pro at zero cost. Fine-tuned SLMs dominate domain-specific context-adherence (NVIDIA 2025, Forbes July 2026).
8. **Thinking hurts scout accuracy** — adds latency with no accuracy gain. Causes DeepSeek to consume all output tokens with internal reasoning.
9. **Prompt engineering is critical** — DeepSeek went from 33% to 73% just by simplifying the output format and reducing reasoning effort.

## Models tested

| preset | model | quant | engine | size | thinking mode |
| ornith | Ornith-1.0-35B-Q6_K-Frankenstein-MTP | Q6_K | llama.cpp | 30 GB | bundled (no hard switch) |
| qwen4 | Qwen3.6-27B-MTP | Q4_K_XL | llama.cpp | 16 GB | disabled via froggeric `<|think_off|>` |
| qwythos | Qwythos-9B-Claude-Mythos-5-MTPLX | MLX 4-bit | MTPLX 2.0.2 | 5 GB | off via `--reasoning off` |
| **mtplx** | **Qwen3.6-27B-MTPLX-Optimized-Speed** | **MLX 4-bit** | **MTPLX 2.0.2** | **15 GB** | **off via `--reasoning off`** |
| thinkingcap | bottlecapAI/ThinkingCap-Qwen3.6-27B | Q4_K_M | llama.cpp | 16 GB | enabled (default) |
| qwen122b | Qwen3.5-122B-A10B-MTP (MoE, 10B active) | UD-Q2_K_XL | llama.cpp | 43 GB | disabled via froggeric `<|think_off|>` |

All llama.cpp Qwen-based models use [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) (v21.3) via `--jinja --chat-template-file`. The MTPLX model uses its built-in `local_qwen36` template with `--reasoning off` (no template hacking needed). Both engines use MTP speculative decoding — llama.cpp at depth 2 (`--spec-type draft-mtp`), MTPLX at depth 3 (native).

## Benchmarks

All three benchmarks use the same random seed (42) so every model sees identical prompts.

### 1. IFEval — Instruction Following (n=100)

- **Dataset**: [google/IFEval](https://huggingface.co/datasets/google/IFEval) (541 prompts), 25 instruction types, strict+loose grading
- **Verifier**: vendored google-research `instruction_following_eval` library, with a kwargs-filter fix (the HF dataset stores all-possible kwargs with nulls for every instruction; the upstream verifier TypeErrors when extras are passed)
- **max_tokens**: 8192

### 2. HumanEval — Function Implementation from Docstring (n=50)

- **Dataset**: [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) (164 problems)
- **Grading**: execution-based — the model's completion is concatenated with hidden test cases and executed via `subprocess.run` with a 10s timeout. Pass = all tests pass without exception.
- **Extraction**: handles prose prefixes and markdown code fences. Falls back to regex-based `def <entry_point>(` extraction.
- **max_tokens**: 2048

### 3. RepoQA — Long-Context Code Retrieval (n=44 or n=50)

- **Dataset**: [evalplus/repoqa](https://github.com/evalplus/repoqa) (500 problems across 5 languages)
- **Task**: Searching Needle Function (SNF) — given a function description and a large code context (full source file, up to 22k tokens), retrieve the exact described function.
- **Grading**: sentence-BLEU (nltk, smoothing method1) between the model's extracted code block and the ground-truth needle function. Pass threshold: ≥0.8. Table reports **avg BLEU** (not pass rate).
- **Note**: the upstream RepoQA pip package is un-installable on Python 3.14 (`tree_sitter_languages` has no cp314 wheel). We vendored the dataset and wrote our own regex-based code-block extractor.
- **max_tokens**: 1024


### 4. Scout Pipeline — Code Retrieval from Large Codebase (n=30)

- **Task**: Given a codebase and a natural-language query, find the exact file, symbol, and line range. Full pipeline: Qwen3-Embedding-0.6B embedder → BGE-Reranker-v2-m3 reranker → scout LLM.
- **Codebase**: MTPLX 2.0.2 source (149 Python files, 11 MB)
- **Queries**: 30 auto-generated queries from function docstrings across 30 different files
- **Grading**: exact file name match (primary), BLEU on extracted code (secondary)
- **max_tokens**: 1024 (local), 2048 (cloud)

## Scout Pipeline Results

| scout model | engine | file match | avg time | cost/run |
|---|---|---|---|---|
| **Qwen3.6-27B MLX 4-bit** | MTPLX | **90.0%** | 138s | $0 |
| Qwythos-9B MLX 4-bit | MTPLX | 86.7% | 40s | $0 |
| DeepSeek v4 Pro (low reasoning) | cloud API | 73.3% | 26s | $0.30 |
| DeepSeek v4 Pro (xhigh reasoning) | cloud API | 33.3% | 65s | $0.21 |

*All models use the same embedder+reranker pipeline. DeepSeek results use `reasoning_effort="low"` — without it, the model spends all tokens on internal reasoning and outputs nothing. xhigh reasoning was worse. The local 27B wins on accuracy; Qwythos 9B is the speed/accuracy sweet spot.*

Key scout findings:

1. **Local models crush cloud on private codebase retrieval** — the 27B (90%) and Qwythos 9B (87%) both beat DeepSeek v4 Pro (73%). Cloud models weren't trained on your code; they hallucinate file paths or ignore the provided context.
2. **Small models can beat giants on narrow tasks** — Qwythos 9B (86.7%) outperforms DeepSeek v4 Pro (73.3%) at 1.5× the speed and zero cost. This matches the industry trend: fine-tuned small models dominate on domain-specific context-adherence tasks (Forbes, July 2026; NVIDIA SLM paper, 2025).
3. **Thinking hurts scout accuracy** — Qwythos think-on scored the same file match (86.7%) as think-off but 2× slower with worse code extraction. Thinking mode also caused DeepSeek to consume all output tokens with internal reasoning.
4. **Prompt engineering matters enormously** — DeepSeek went from 33% (verbose format + xhigh) to 73% (simple "which file?" + low reasoning). The original format buried instructions in a wall of code.
5. **The embedder matters less than the scout LLM** — Qwen3-Embedding-0.6B (MTEB #1 for 0.6B class) + BGE-Reranker-v2-m3 is a SOTA stack. The bottleneck is consistently the scout LLM's ability to read provided context and follow instructions.

## Quick start

```bash
git clone https://github.com/bradley-mankoff/llm-gauntlet.git
cd llm-gauntlet

# Install dependencies (requires uv)
uv sync

# --- llama.cpp path ---
# Serve qwen4 (requires llama.cpp: brew install llama.cpp)
BENCH=1 MODEL_PRESET=qwen4 USE_FROGGERIC_CHAT_TEMPLATE=1 ./scripts/serve.sh

# Run a benchmark
uv run python bench_one.py \
    --benchmark humaneval --model auto \
    --out results/humaneval_qwen4_n50.json \
    --n-samples 50 --max-tokens 2048 --seed 42 --think-off

# --- MTPLX path ---
# Serve MTPLX model (brew install youssofal/mtplx/mtplx)
mtplx serve --model Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed \
    --port 8000 --reasoning off --no-stats-footer --profile turbo

# Run a benchmark (point at MTPLX's port)
uv run python bench_one.py \
    --benchmark ifeval --model auto \
    --base-url http://localhost:8000/v1 \
    --out results/ifeval_mtplx_n100.json \
    --n-samples 100 --max-tokens 8192 --seed 42

# Full gauntlet (llama.cpp models only)
N_SAMPLES=50 MAX_TOKENS_IFEVAL=8192 ./run-gauntlet.sh
```

## Repo structure

```
├── README.md
├── LICENSE                     # Apache 2.0
├── pyproject.toml              # uv project config
├── bench_one.py                # runner: one model × one benchmark → JSON
├── bench_client.py             # thin OpenAI SDK wrapper (llama-server / MTPLX)
├── run-gauntlet.sh             # full gauntlet driver
├── chat_template.jinja         # froggeric/Qwen-Fixed-Chat-Templates v21.3
├── scripts/
│   └── serve.sh                # reference serving script
├── tasks/
│   ├── ifeval.py               # IFEval task with kwargs-filter fix
│   ├── humaneval.py            # HumanEval task with execution-based grading
│   └── repoqa.py               # RepoQA task with BLEU grading
├── ifeval_lib/
│   └── instruction_following_eval/  # vendored google-research verifier (Apache 2.0)
├── results/                    # JSON outputs land here (gitignored)
└── logs/                       # per-run logs (gitignored)
```

## Known limitations

- RepoQA grading is BLEU-based with regex code-block extraction (not tree-sitter-syntactic).
- HumanEval grading uses `subprocess.run` (not a sandbox). Test timeout is 10s.
- IFEval uses the google-research verifier with a kwargs-filter compatibility layer.
- All models use temperature=0 (greedy decoding). Sampling variance is not measured.
- The MTPLX turbo profile is unstable under sustained reasoning load on M1 Max (crashes after 10-38 samples). Use `--profile sustained` for thinking-on benchmarks, or `--profile turbo` only for thinking-off.
- Qwen5 was purged from disk (Q5+Q6 made negligible difference over Q4).

## Citation / sources

- **IFEval**: Zhou et al., "Instruction-Following Evaluation for Large Language Models", 2023. `google/IFEval` on HuggingFace. Verifier: `google-research/instruction_following_eval` (Apache 2.0, vendored).
- **HumanEval**: Chen et al., "Evaluating Large Language Models Trained on Code", 2021. `openai/openai_humaneval` on HuggingFace.
- **RepoQA**: Tian et al., "RepoQA: Evaluating Long-Context Code Understanding", ICML 2024. `evalplus/repoqa_release`.
- **Chat template**: froggeric/Qwen-Fixed-Chat-Templates, v21.3, Apache 2.0.
- **MTPLX**: Youssof Altoukhi, "MTPLX — Native MTP Speculative Decoding on Apple Silicon", 2026. `github.com/youssofal/MTPLX`.
- **Models**: Ornith-1.0 (DeepReinforce AI), Qwen3.6-27B-MTP (Alibaba/unsloth), ThinkingCap (BottleCap AI), Qwen3.5-122B-A10B (Alibaba/unsloth). MTPLX-optimized Qwen3.6-27B (Youssofal).
## License

Apache 2.0. The vendored IFEval verifier (`ifeval_lib/`) is also Apache 2.0 from
google-research. The froggeric chat template is Apache 2.0.
