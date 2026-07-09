# Local LLM Benchmark Results

**Date**: 2026-07-06 – 2026-07-09  
**Hardware**: MacBook Pro M1 Max, 64 GB unified memory, 10 cores, macOS 26.5.1  
**Inference**: llama.cpp 9840 (8c146a836), Apple Metal, kv-unified q8_0 cache, 262k ctx, np=1  
**Benchmark harness**: `${HOME}/llama-runs/gauntlet/` (uv-managed Python project; see `benchmarker_instructions.md` for the full setup)

## Models tested

| preset | model | quant | size | thinking mode |
|---|---|---|---|---|
| ornith | Ornith-1.0-35B-Q6_K-Frankenstein-MTP | Q6_K | 30 GB | bundled (no hard switch) |
| qwen4 | Qwen3.6-27B-MTP | Q4_K_XL | 16 GB | disabled via froggeric `` |
| qwen6 | Qwen3.6-27B-MTP | Q6_K_XL | 23 GB | disabled via froggeric `` |
| thinkingcap | bottlecapAI/ThinkingCap-Qwen3.6-27B | Q4_K_M | 16 GB | enabled (default, no override) |
| qwen122b | Qwen3.5-122B-A10B-MTP (MoE, 10B active) | UD-Q2_K_XL | 43 GB | disabled via froggeric `` |

- Ornith is a Frankenstein merge. All Qwen-based models used the [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) (v21.3) via `--jinja --chat-template-file`. Thinking on/of is controlled with the froggeric `` tag injected into user messages at the bench layer.
- The upstream Qwen3.6 chat template has a known infinite-thinking-loop bug; froggeric cured it and also provided the `` hard switch that llama.cpp's built-in `--chat-template-kwargs` does not expose (per qwen.readthedocs.io and llama.cpp issue 20409).

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

| model | IFEval strict | s/sample | HumanEval pass@1 | s/sample | RepoQA avg BLEU | s/sample | thinking |
|---|---|---|---|---|---|---|---|
| Ornith-35B-Q6_K | 73% | 102 | 46% | 117 | 0.259 | 53 | on (bundled) |
| Qwen3.6-27B Q4_K_XL | 79% | 37 | **88%** | 28 | 0.513 | — | off |
| Qwen3.6-27B Q6_K_XL | 80% | 40 | **88%** | 25 | — | — | off |
| ThinkingCap Q4_K_M | **86%** | 209 | 74% | 145 | **0.779** | 164 | on |
| Qwen3.5-122B-A10B Q2_K_XL | 83% | 38 | 84% | 49 | 0.219 | 57 | off |

*Qwen4 RepoQA wall time missing (run killed mid-stream; 44/50 tasks completed). Qwen6 RepoQA data missing (run cancelled before first result).*

## Key findings

1. **ThinkingCap wins on accuracy** — 86% IFEval, 0.779 RepoQA BLEU — but at a 4–6× wall-time premium. Its RL-thought-control fine-tune pays off in quality but doesn't reduce wall time vs vanilla Qwen with thinking off.
2. **Qwen4/Qwen6 with think_off are the speed winners** — 88% HumanEval at 25–28 s/sample. Both the IFEval and HumanEval scores are GPT-4 territory (GPT-4 scored ~76% on IFEval in the published paper).
3. **The 122B MoE at 2-bit surprises** — 83% IFEval and 84% HumanEval at 38–49 s/sample is strong for a UD_Q2_K_XL model costing 43 GB. But RepoQA at 0.219 BLEU shows that the 2-bit quantization severely degrades long-context retrieval.
4. **Q4→Q6 buys nothing** — 79%→80% IFEval, 88%→88% HumanEval. The quantization step is within noise for n=100/50.
5. **The froggeric chat template + `＜|think_off|＞` combo is the production recipe for Qwen on Apple Metal** — it's the only working hard switch for thinking, and it turns a 17-hour impractical bench into a 1-hour practical one.
6. **Ornith trails across the board** — last on HumanEval and RepoQA, second on IFEval. The only model still running with bundled thinking.

## Known limitations

- RepoQA grading is BLEU-based with regex code-block extraction (not tree-sitter-syntactic). Models that wrap their answer in a single ``` block get full credit.
- HumanEval grading uses `subprocess.run` (not a sandbox). The test timeout is 10s.
- IFEval uses the google-research verifier with a kwargs-filter compatibility layer. The underlying instruction checkers are the published ones.
- All models use temperature=0 (greedy decoding). Sampling variance is not measured.
- The Qwen3.6-27B models serve via `--hf-repo`/`--hf-file` in llama-server, which uses the unsloth GGUF quantizations from HuggingFace.
- Qwen5 and Qwen6 were purged from disk (Q5+Q6 made negligible difference over Q4 per the data).

## File manifest

- `${HOME}/llama-runs/gauntlet/benchmarker_instructions.md` — full setup, how to reproduce, path conventions, gotchas
- `${HOME}/llama-runs/gauntlet/tasks/ifeval.py` — IFEval task with kwargs-filter fix
- `${HOME}/llama-runs/gauntlet/tasks/humaneval.py` — HumanEval task with execution-based grading
- `${HOME}/llama-runs/gauntlet/tasks/repoqa.py` — RepoQA task with BLEU grading
- `${HOME}/llama-runs/gauntlet/ifeval_lib/` — vendored google-research IFEval verifier
- `${HOME}/llama-runs/chat_template.jinja` — froggeric/Qwen-Fixed-Chat-Templates v21.3
- `${HOME}/llama-runs/serve-up.sh` — server launcher with model presets
- `${HOME}/llama-runs/gauntlet/run-gauntlet.sh` — full gauntlet driver

## To reproduce

```bash
# serve a model
cd ~/llama-runs && BENCH=1 MODEL_PRESET=qwen4 USE_FROGGERIC_CHAT_TEMPLATE=1 ./serve-up.sh

# run a single benchmark
cd ~/llama-runs/gauntlet && uv run python bench_one.py \
    --benchmark humaneval --model auto \
    --out results/humaneval_qwen4_n50_random.json \
    --n-samples 50 --max-tokens 2048 --seed 42 --think-off

# full gauntlet (2 models × 3 benchmarks)
cd ~/llama-runs/gauntlet && N_SAMPLES=50 MAX_TOKENS_IFEVAL=8192 ./run-gauntlet.sh
```

## Citation / sources

- **IFEval**: Zhou et al., "Instruction-Following Evaluation for Large Language Models", 2023. Dataset: `google/IFEval` on HuggingFace. Verifier: `google-research/instruction_following_eval` (Apache 2.0, vendored).
- **HumanEval**: Chen et al., "Evaluating Large Language Models Trained on Code", 2021. Dataset: `openai/openai_humaneval` on HuggingFace.
- **RepoQA**: Tian et al., "RepoQA: Evaluating Long-Context Code Understanding", ICML 2024. Dataset: `evalplus/repoqa_release` (2024-06-23, 50 repos × 10 needles × 5 languages = 500 tasks).
- **Chat template**: froggeric/Qwen-Fixed-Chat-Templates, v21.3, Apache 2.0.
- **Models**: Ornith-1.0 (DeepReinforce AI), Qwen3.6-27B-MTP (Alibaba/unsloth), ThinkingCap (BottleCap AI), Qwen3.5-122B-A10B (Alibaba/unsloth). All served via unsloth GGUF quantizations on HuggingFace.
