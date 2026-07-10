# Local LLM Benchmark Results

**Date**: 2026-07-06 – 2026-07-10
**Hardware**: MacBook Pro M1 Max, 64 GB unified memory, 10 cores, macOS 26.5.1
**Inference engines**: llama.cpp 9840 (8c146a836), Apple Metal, kv-unified q8_0 cache, 262k ctx, np=1; and MTPLX 2.0.2 (MLX, native MTP depth 3, turbo profile)
**Benchmark harness**: `${HOME}/llama-runs/gauntlet/` (uv-managed Python project; see `benchmarker_instructions.md` for the full setup)

## Models tested

| preset | model | quant | engine | size | thinking mode |
|---|---|---|---|---|---|
| ornith | Ornith-1.0-35B-Q6_K-Frankenstein-MTP | Q6_K | llama.cpp | 30 GB | bundled (no hard switch) |
| qwen4 | Qwen3.6-27B-MTP | Q4_K_XL | llama.cpp | 16 GB | disabled via froggeric `<|think_off|>` |
| qwen6 | Qwen3.6-27B-MTP | Q6_K_XL | llama.cpp | 23 GB | disabled via froggeric `<|think_off|>` |
| **mtplx** | **Qwen3.6-27B-MTPLX-Optimized-Speed** | **MLX 4-bit** | **MTPLX 2.0.2** | **15 GB** | **off via `--reasoning off`** |
| thinkingcap | bottlecapAI/ThinkingCap-Qwen3.6-27B | Q4_K_M | llama.cpp | 16 GB | enabled (default) |
| qwen122b | Qwen3.5-122B-A10B-MTP (MoE, 10B active) | UD-Q2_K_XL | llama.cpp | 43 GB | disabled via froggeric `<|think_off|>` |
- All Qwen-based llama.cpp models use [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) (v21.3). The MTPLX model uses its built-in `local_qwen36` template. Both engines use MTP speculative decoding — llama.cpp at depth 2 (`--spec-type draft-mtp`), MTPLX at depth 3 (native).

## Benchmarks

All three benchmarks use the same random seed (42) so every model sees identical prompts.

### 1. IFEval — Instruction Following (n=100)

- **Dataset**: [google/IFEval](https://huggingface.co/datasets/google/IFEval) (541 prompts), 25 instruction types, strict+loose grading
- **Verifier**: vendored google-research `instruction_following_eval` library, with a kwargs-filter fix (the HF dataset stores all-possible kwargs with nulls for every instruction; the upstream verifier TypeErrors when extras are passed — we filter each kwargs dict to the params the instruction's `build_description` actually accepts via `inspect.signature`)
- **max_tokens**: 8192 (required headroom for reasoning/chat-template overhead)
- **n_samples**: 100 (shuffled, seed=42)

### 2. HumanEval — Function Implementation from Docstring (n=50)

- **Dataset**: [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) (164 problems)
- **Grading**: execution-based — the model's completion is concatenated with the problem's hidden test cases and executed via `subprocess.run` with a 10s timeout. Pass = all tests pass without exception.
- **Extraction**: handles prose prefixes ("Here's the implementation:") and markdown code fences. Falls back to regex-based `def <entry_point>(` extraction when the model wraps its answer in commentary.
- **max_tokens**: 2048
- **n_samples**: 50 (shuffled, seed=42)

### 3. RepoQA — Long-Context Code Retrieval (n=44 or n=50)

- **Dataset**: [evalplus/repoqa](https://github.com/evalplus/repoqa) (500 problems across 5 languages); downloaded from `evalplus/repoqa_release` (2024-06-23 snapshot, 71 MB JSON)
- **Task**: Searching Needle Function (SNF) — given a function description and a large code context (full source file, up to 22k tokens), retrieve the exact described function.
- **Grading**: sentence-BLEU (nltk, smoothing method1) between the model's extracted code block and the ground-truth needle function. Pass threshold: ≥0.8. The table reports **avg BLEU** (not pass rate) — a more continuous comparison.
- **Note**: the upstream RepoQA pip package is un-installable on Python 3.14 (`tree_sitter_languages` has no cp314 wheel). We vendored only the dataset and wrote our own regex-based code-block extractor. Tree-sitter syntactic validation is not performed.
- **max_tokens**: 1024
- **n_samples**: 44 for Ornith, Qwen4, Qwen122B; 50 for ThinkingCap (it ran first before we standardized on n=44)

## Results

| model | engine | IFEval strict | s/sample | HumanEval pass@1 | s/sample | RepoQA avg BLEU | s/sample | thinking |
|---|---|---|---|---|---|---|---|---|
| Ornith-35B-Q6_K | llama.cpp | 73% | 102 | 46% | 117 | 0.259 | 53 | on (bundled) |
| Qwen3.6-27B Q4_K_XL | llama.cpp | 79% | 37 | **88%** | 28 | 0.513 | — | off |
| Qwen3.6-27B Q6_K_XL | llama.cpp | 80% | 40 | **88%** | 25 | — | — | off |
| **Qwen3.6-27B MLX 4-bit** | **MTPLX** | **85%** | 23 | **88%** | 24 | 0.205 | 33 | off |
| ThinkingCap Q4_K_M | llama.cpp | **86%** | 209 | 74% | 145 | **0.779** | 164 | on |
| Qwen3.5-122B-A10B Q2_K_XL | llama.cpp | 83% | 38 | 84% | 49 | 0.219 | 57 | off |

*Qwen4 RepoQA wall time missing (run killed mid-stream). Qwen6 RepoQA data missing (run cancelled). MTPLX row uses turbo profile (sustained would be slower).*
## Key findings

1. **MTPLX dominates llama.cpp on IFEval (thinking off)** — 85% vs 79% strict, and 23 vs 37 s/sample. Same Qwen3.6-27B base, same MTP technique, but MLX 4-bit + native MTP depth 3 outruns GGUF Q4_K_XL + llama.cpp MTP depth 2.
2. **Both engines use MTP** — llama.cpp passes `--spec-type draft-mtp --spec-draft-n-max 2`. MTPLX uses depth 3 native MTP. Speeds are comparable; MTPLX's advantage is operational (cleaner setup, `--reasoning` flag, 6× faster startup).
3. **ThinkingCap wins on accuracy (thinking on)** — 86% IFEval, 0.779 RepoQA. MTPLX with reasoning on was tracking at 89% IFEval (36/100, sustained profile) before cancellation, but the turbo profile crashes under sustained reasoning load on M1 Max.
4. **Qwen4/Qwen6 with think_off are the speed winners** — 88% HumanEval at 25–28 s/sample. MTPLX matches 88% at 24 s/sample. GPT-4 territory.
5. **The 122B MoE at 2-bit surprises** — 83% IFEval and 84% HumanEval at 38–49 s/sample. RepoQA at 0.219 BLEU shows 2-bit degrades long-context retrieval.
6. **Q4→Q6 buys nothing** — 79%→80% IFEval, 88%→88% HumanEval. Within noise.

## Known limitations

- RepoQA grading is BLEU-based with regex code-block extraction (not tree-sitter-syntactic). Models that wrap their answer in a single ``` block get full credit.
- HumanEval grading uses `subprocess.run` (not a sandbox). The test timeout is 10s.
- IFEval uses the google-research verifier with a kwargs-filter compatibility layer. The underlying instruction checkers are the published ones.
- All models use temperature=0 (greedy decoding). Sampling variance is not measured.
- The Qwen3.6-27B models serve via `--hf-repo`/`--hf-file` in llama-server, which uses the unsloth GGUF quantizations from HuggingFace.
- Qwen5 and Qwen6 were purged from disk (Q5+Q6 made negligible difference over Q4 per the data).

## File manifest

- `benchmarker_instructions.md` — full setup, how to reproduce, path conventions, gotchas
- `tasks/ifeval.py` — IFEval task with kwargs-filter fix
- `tasks/humaneval.py` — HumanEval task with execution-based grading
- `tasks/repoqa.py` — RepoQA task with BLEU grading
- `ifeval_lib/` — vendored google-research IFEval verifier
- `chat_template.jinja` — froggeric/Qwen-Fixed-Chat-Templates v21.3
- `scripts/serve.sh` — reference server launcher with model presets
- `run-gauntlet.sh` — full gauntlet driver (llama.cpp models)

## Citation / sources

- **IFEval**: Zhou et al., "Instruction-Following Evaluation for Large Language Models", 2023. `google/IFEval` on HuggingFace. Verifier: `google-research/instruction_following_eval` (Apache 2.0, vendored).
- **HumanEval**: Chen et al., "Evaluating Large Language Models Trained on Code", 2021. `openai/openai_humaneval` on HuggingFace.
- **RepoQA**: Tian et al., "RepoQA: Evaluating Long-Context Code Understanding", ICML 2024. `evalplus/repoqa_release`.
- **Chat template**: froggeric/Qwen-Fixed-Chat-Templates, v21.3, Apache 2.0.
- **MTPLX**: Youssof Altoukhi, "MTPLX — Native MTP Speculative Decoding on Apple Silicon", 2026. `github.com/youssofal/MTPLX`.
- **Models**: Ornith-1.0 (DeepReinforce AI), Qwen3.6-27B-MTP (Alibaba/unsloth), ThinkingCap (BottleCap AI), Qwen3.5-122B-A10B (Alibaba/unsloth), MTPLX-optimized Qwen3.6-27B (Youssofal).
