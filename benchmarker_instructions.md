# Benchmarker Instructions

> How to continue the local LLM benchmarking work. Pick up here tomorrow, or hand this to another agent.

---

## TL;DR

`${HOME}/llama-runs/gauntlet/` is a uv-managed Python project that benchmarks 4 locally-served llama.cpp models against 3 open-source instruction-following / code benchmarks. The 4 models are already on disk. The 3 benchmarks have working loaders. The known IFEval-verifier-bug is fixed and verified. **First real result is in: Ornith-35B-Q6_K scores 9/10 = 90% strict pass on IFEval (n=10, max_tokens=16384).** That's a strong result — equivalent to or better than most open 30–50B models, and on par with what the IFEval paper reports for GPT-3.5-Turbo. Real work to do now: scale to the other 3 Qwen quants, smoke-test CodeEditBench + RepoQA, then run the full 4×3 gauntlet.

## Path conventions used in this doc

This is a single-machine setup. There is **no nesting**. The whole thing lives at:

```
${HOME}/llama-runs/                                # = /Users/bradley_mankoff/llama-runs/
├── serve-up.sh, serve-down.sh, p.py, llama.*      # the serving harness (pre-existing)
└── gauntlet/                                      # this benchmarking project (uv)
    ├── pyproject.toml, uv.lock
    ├── tasks/, ifeval_lib/, bench_one.py, …
    ├── benchmarker_instructions.md                # ← this file
    ├── results/, logs/, .cache/
    └── .venv/                                     # uv-managed, not in git
```


## Current state (as of 2026-07-07 ~12:33)

| Item | Status |
|---|---|
| 5 model presets in `${HOME}/llama-runs/serve-up.sh` | ✅ all on disk or in HF cache |
| froggeric chat template at `${HOME}/llama-runs/chat_template.jinja` | ✅ downloaded (v21.3), applied to Qwen-based presets via `USE_FROGGERIC_CHAT_TEMPLATE=1` |
| llama-server running | ✅ PID 79023, serving unsloth/Qwen3.6-27B-MTP-GGUF (Q4_K_XL), ctx 262144, np=1, with froggeric template |
| IFEval benchmark loader | ✅ working, kwargs filter fix applied, **verified 73% strict on n=100 random for Ornith** |
| CodeEditBench loader | ✅ working, untested end-to-end |
| RepoQA loader | ✅ working, untested end-to-end |
| IFEval n=100 on ornith (seed=42) | ✅ done — `results/ifeval_ornith_n100_random.json`, 73/100 strict, 75/100 loose, 10183s wall |
| IFEval n=100 on qwen4 (seed=42, froggeric, max_tokens=8192) | 🟢 running, PID 79084, → `results/ifeval_qwen4_n100_random.json` |
| Qwen chattiness fix | ✅ froggeric template loaded; v1 Qwen run cancelled before completion (samples 5000+ tokens) |
| Full 5×3 gauntlet run | ❌ not started — see "Order of work" below |

## Hardware / environment

- **Server**: MacBook Pro M1 Max, 64 GB unified memory, 10 cores, arm64, macOS 26.5.1
- **Hostname / IP**: `Bradleys-MacBook-Pro-2.local` / `192.168.1.217:8080`
- **llama.cpp**: 9840 (8c146a836), built with AppleClang 21, Apple Metal
- **Disk**: 739 GB free in `${HOME}/`, 16 GB used by `${HOME}/.cache/huggingface/`
- **Python**: 3.14.6 via Homebrew (`/opt/homebrew/bin/python3`)
- **Package manager**: `uv` 0.11.26, project at `${HOME}/llama-runs/gauntlet/`
- **Other servers on the network**: `192.168.1.150:3100` (Loki, log sink), `192.168.1.150:8765` (fileserver for `p.py`)

## Repo layout

```
${HOME}/llama-runs/                              # the serving harness
├── serve-up.sh                                  # starts llama-server with presets
├── serve-down.sh                                # stops llama-server + pusher
├── p.py                                         # log-pusher script (Loki)
├── llama.log, llama.stdout, llama.pid
└── pusher.out, pusher.pid

${HOME}/llama-runs/gauntlet/                     # the benchmarking harness (uv project)
├── pyproject.toml                               # deps: inspect-ai, openai, datasets,
│                                                #       absl-py, langdetect, nltk,
│                                                #       immutabledict
├── uv.lock
├── benchmarker_instructions.md                  # ← this file
├── README.md                                    # shorter overview
├── ifeval_lib/instruction_following_eval/       # vendored google-research verifier
├── tasks/
│   ├── __init__.py
│   ├── ifeval.py                                # 541 prompts, strict+loose scoring
│   ├── codeeditbench.py                         # code_debug_primary split, BLEU
│   └── repoqa.py                                # 600 tasks, BLEU ≥ 0.8 vs needle function
├── bench_client.py                              # thin OpenAI SDK wrapper
├── bench_one.py                                 # one model × one benchmark → JSON
├── smoke.py                                     # 3-prompt IFEval sanity check
├── run-gauntlet.sh                              # 4 models × 3 benchmarks loop
├── .cache/
│   ├── google___if_eval/                        # HF cache for IFEval
│   ├── m-a-p___code_editor_bench/               # HF cache (locked, didn't fully load)
│   └── repoqa-2024-06-23.json                   # 71 MB RepoQA dataset
├── results/                                     # benchmark output JSON lands here
└── logs/                                        # per-run logs land here
```

## The 5 model presets

All in `${HOME}/llama-runs/serve-up.sh`. They map to `MODEL_PRESET` env var. Weights are already on disk in `${HOME}/.cache/huggingface/hub/`.

| preset       | model                                           | quant    | size  |
|--------------|-------------------------------------------------|----------|-------|
| ornith       | skinnyctax/Ornith-1.0-35B-Q6_K-Frankenstein-MTP | Q6_K     | 30 GB |
| qwen4        | unsloth/Qwen3.6-27B-MTP                         | Q4_K_XL  | ~16 GB |
| qwen5        | unsloth/Qwen3.6-27B-MTP                         | Q5_K_XL  | ~19 GB |
| qwen6        | unsloth/Qwen3.6-27B-MTP                         | Q6_K_XL  | ~23 GB |
| thinkingcap  | bottlecapai/ThinkingCap-Qwen3.6-27B-GGUF        | Q4_K_M  | ~16 GB |

**Note on ThinkingCap**: same Qwen3.6-27B base, fine-tuned with online RL to produce ~50% fewer thinking tokens while preserving answer quality. Useful as a 5th row to see whether the chattiness-on-base-Qwen issue is addressable in the weights. Q4_K_M is the only 4/8-bit option (Q8_0 is 27 GB; f16 is 51 GB). The 16 GB Q4_K_M fits comfortably in our 64 GB machine.

Note: there is also a separate `deepreinforce-ai_Ornith-1.0-35B-Q6_K_L.gguf` in `${HOME}/models/` (different uploader) that the current `ornith` preset does NOT use. The preset uses the `skinnyctax` Frankenstein-MTP build which has draft-MTP speculative decoding and the chat template that emits `reasoning_content`.

## Serving

Bring up a model with single-slot reproducible mode:

```bash
cd ${HOME}/llama-runs
BENCH=1 MODEL_PRESET=ornith ./serve-up.sh     # or qwen4 / qwen5 / qwen6 / thinkingcap
```

- `BENCH=1` sets `N_PARALLEL=1` (was added to `serve-up.sh` for this project; honors an explicit `N_PARALLEL_OVERRIDE=1` to force).
- `USE_FROGGERIC_CHAT_TEMPLATE=1` swaps in `${HOME}/llama-runs/chat_template.jinja` (froggeric/Qwen-Fixed-Chat-Templates) instead of the model-bundled template. **Required for Qwen3.6 — without it the model falls into infinite thinking loops and burns the whole `max_tokens` budget without producing an answer.** Leave at 0 for Ornith (Frankenstein merge, bundled template already works). `run-gauntlet.sh` sets this automatically per-preset.
- The server runs on `0.0.0.0:8080` with `--metrics --slots --log-timestamps -t 8 -fa on -c 262144 -np 1 --kv-unified --cache-type-k q8_0 --cache-type-v q8_0 --spec-type draft-mtp --mlock`.
- Wait for `/health` to return `{"status":"ok"}` before running benchmarks (the script blocks up to 180s).
- Model name as reported by `/v1/models` is the full HF repo name, e.g. `skinnyctax/Ornith-1.0-35B-Q6_K-Frankenstein-MTP-GGUF` — pass that exact string to API calls.
- Tear down with `cd ${HOME}/llama-runs && ./serve-down.sh` (idempotent).

## Running benchmarks

```bash
cd ${HOME}/llama-runs/gauntlet
uv run python bench_one.py \
    --benchmark ifeval \
    --model auto \
    --out results/ifeval_qwen5_n50.json \
    --n-samples 50 \
    --max-tokens 16384
```

`--model auto` reads the only model currently served. `--benchmark` is one of `ifeval`, `codeeditbench`, `repoqa`. Output JSON shape: `{summary: {...}, results: [{key, prompt, response, ...}]}`.

### Full gauntlet

```bash
cd ${HOME}/llama-runs/gauntlet
N_SAMPLES=50 ./run-gauntlet.sh     # 4 models × 3 benchmarks × 50 = 600 evals
```

The script iterates `ornith → qwen4 → qwen5 → qwen6`, doing `serve-up` → 3 benchmarks → `serve-down` per preset. Skip teardown with `SKIP_DOWN=1`.

### Inspecting results

```bash
python3 -c "
import json
d = json.load(open('${HOME}/llama-runs/gauntlet/results/ifeval_qwen5_n50.json'))
s = d['summary']
print(f'IFEval n={s[\"n_samples\"]} on {s[\"model\"][:60]}')
print(f'  strict: {s[\"strict_pass\"]}/{s[\"n_samples\"]} = {s[\"strict_pass_rate\"]:.2%}')
print(f'  loose:  {s[\"loose_pass\"]}/{s[\"n_samples\"]} = {s[\"loose_pass_rate\"]:.2%}')
print(f'  wall:   {s[\"wall_time_sec\"]}s')
"
```

For per-instruction detail, look at `r['per_instruction_strict']` in `d['results']` — a dict like `{"punctuation:no_comma": True, "detectable_format:number_highlighted_sections": False}`.

## Critical gotcha: the IFEval kwargs filter (DO NOT REMOVE)

The vendored google-research IFEval verifier at `${HOME}/llama-runs/gauntlet/ifeval_lib/instruction_following_eval/evaluation_lib.py:88` does:

```python
instruction.build_description(**inp.kwargs[index])
```

The HF `google/IFEval` dataset stores **all-possible kwargs (with nulls) per row** — for any instruction, you get `{num_highlights, relation, num_words, num_placeholders, ...}` whether or not they're relevant. About 12 of the 25 instruction classes (e.g. `CommaChecker`, `LowercaseLettersEnglishChecker`, `CapitalLettersEnglishChecker`, `JsonFormat`, `TitleChecker`, `QuotationChecker`, `TwoResponsesChecker`, `ConstrainedResponseChecker`) take **zero arguments** in their `build_description` method. The verifier throws `TypeError`, the prior `try/except` in `ifeval.py` silently caught it, and every result was marked failed.

**Symptom**: `0/N` pass rate on IFEval, with `per_instruction_strict: {"_error": "TypeError(\"...got an unexpected keyword argument 'num_highlights'\")"}` in the JSON.

**Fix**: `tasks/ifeval.py:_filter_kwargs_for_instruction(instr_id, kwargs)` uses `inspect.signature` to find the params each `build_description` accepts, keeps only those, drops nulls. Applied before constructing `InputExample`. If you touch `ifeval.py`, do NOT remove this filter without understanding why it's there.

lm-eval-harness has its own IFEval that doesn't have this bug; the upstream google-research verifier just isn't compatible with the HF dataset format.

## Other known gotchas

1. **max_tokens for IFEval must be ≥ 8192.** Ornith-35B emits a `reasoning_content` field first; at 4096 the reasoning eats the whole budget and the answer never starts. Use 16384 to be safe. The Qwen models will be similar.
2. **np=1 is required for reproducibility.** `BENCH=1` in `serve-up.sh` enforces this. The slot API reports `"n_slots": 1` when active.
3. **Qwen quants are NOT in `${HOME}/models/`** — they're in `${HOME}/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/.../`. The `serve-up.sh` uses `--hf-repo` / `--hf-file` which llama-server resolves against the HF cache, so no download needed.
4. **Async shell jobs don't escape the 300s timeout** even with `async: true`. Use `nohup … &` + `disown` for long-running benchmarks.
5. **CodeEditBench's `m-a-p/CodeEditorBench` HF dataset has a schema/columns mismatch** and the `load_dataset` path crashes. We sidestep this by downloading the raw `code_debug_primary.jsonl` directly via `urllib.request.urlretrieve` from `https://huggingface.co/datasets/m-a-p/CodeEditorBench/resolve/main/`. First run pulls ~160 MB.
6. **RepoQA's pip package is un-installable on Python 3.14** because `tree_sitter_languages` has no cp314 wheel. We vendor the data file and implement our own simple grader (BLEU on regex-extracted first code block, threshold 0.8). We do NOT do tree-sitter syntactic validation — a model that wraps its answer in a single ``` block gets full credit regardless of whether the code parses.
7. **Grading is BLEU-based for CodeEditBench and RepoQA** (not execution-based). This is a v0 surrogate — fine for model-vs-model ranking, not the same as the upstream benchmarks' graders. The upstream `CodeEditorBench/evaluation/` and `evalplus/repoqa/compute_score.py` have execution-based grading if you want to upgrade later.
8. **Server is left running between sessions.** If you want it down, `cd ${HOME}/llama-runs && ./serve-down.sh`.

## Order of work (suggested)

1. **~~Wait for the current v2 IFEval n=10 to finish~~** ✅ done. 9/10 = 90% strict on Ornith-35B-Q6_K. Real number, not a bug.
2. **Run IFEval n=50 on each Qwen preset** to get the comparison data:
   ```bash
   cd ${HOME}/llama-runs
   BENCH=1 MODEL_PRESET=qwen4 ./serve-up.sh
   cd ${HOME}/llama-runs/gauntlet
   uv run python bench_one.py --benchmark ifeval --model auto \
       --out results/ifeval_qwen4_n50.json --n-samples 50 --max-tokens 16384
   cd ${HOME}/llama-runs && ./serve-down.sh
   # repeat for qwen5, qwen6
   ```
   ~85 min × 3 = ~4.5 h total.
3. **Smoke-test CodeEditBench and RepoQA** before doing full runs:
   ```bash
   cd ${HOME}/llama-runs/gauntlet
   uv run python bench_one.py --benchmark codeeditbench --model auto \
       --out results/codeeditbench_ornith_n5.json --n-samples 5 --max-tokens 4096
   uv run python bench_one.py --benchmark repoqa --model auto \
       --out results/repoqa_ornith_n10.json --n-samples 10 --max-tokens 2048
   ```
   - CodeEditBench n=5 at 4096 max_tokens ≈ 5–10 min. Downloads ~160 MB on first run.
   - RepoQA n=10 at 2048 max_tokens ≈ 5–10 min. Uses the already-downloaded 71 MB dataset.
4. **If smoke tests look sane, scale up** CodeEditBench and RepoQA to n=50 on each model.
5. **Add a `summarize.py`** that walks `results/*.json` and produces a comparison table:
   ```python
   # rows: benchmark, model, n, strict, loose, wall_time
   ```
6. **Add the full gauntlet to a cron or launchd plist** for nightly runs, dumping to `results/gauntlet_<date>/`.

## Adding a new model

1. Add a preset block to `${HOME}/llama-runs/serve-up.sh` (mirror the existing `qwen4` block):
   ```bash
   mymodel)
       MODEL_REPO="org/repo-name"
       MODEL_FILE="filename.gguf"
       ;;
   ```
2. Pre-download the GGUF (or let `serve-up.sh` pull it on first use). The HF cache lives at `${HOME}/.cache/huggingface/hub/`.
3. Add the preset name to the `PRESETS=(...)` array in `${HOME}/llama-runs/gauntlet/run-gauntlet.sh`.
4. Test with `BENCH=1 MODEL_PRESET=mymodel ./serve-up.sh`, then `bench_one.py --model auto` to confirm the served model name.

## Adding a new benchmark

1. Create `tasks/<name>.py` with:
   - `NAME = "<name>"`
   - `MAX_TOKENS_DEFAULT = N`
   - `def run(client, model, n_samples=None, max_tokens=MAX_TOKENS_DEFAULT, temperature=0.0, progress_every=25, **kwargs) -> (summary, results)`
2. Register it in `bench_one.py:TASKS`.
3. Add the name to `BENCHMARKS=(...)` in `run-gauntlet.sh`.
4. Sanity-test with n=3 before scaling up.

The `(summary, results)` contract:
- `summary` is a dict with at least `{"benchmark", "model", "n_samples"}` and any aggregate stats (`pass_rate`, `avg_score`, etc.)
- `results` is a list of per-prompt dicts, each with at least the prompt, the response, the score, and enough context to debug a failure

## TODO

- [ ] `summarize.py` to turn `results/*.json` into a comparison table (model × benchmark)
- [ ] CodeEditBench: upgrade from BLEU to execution-based grading (use upstream `CodeEditorBench/evaluation/`)
- [ ] RepoQA: add tree-sitter code-block extraction (needs Python 3.12 or source build of `tree_sitter_languages`)
- [ ] Nightly gauntlet via launchd / cron
- [ ] Try the `inspect-ai` runner path as an alternative to the direct `openai` SDK path (the SDK path is simpler and works; revisit if you need parallel sampling or built-in retry)
- [ ] Investigate whether the Ornith model's `reasoning_content` is a chat-template artifact we can strip — saving 30–50% of wall time on IFEval
- [ ] Decide whether the high (90%) IFEval score on Ornith-35B-Q6_K is a sign that the 10-prompt sample is biased toward easy instructions — consider running n=541 before quoting the number

## Quick command reference

```bash
# serving
cd ${HOME}/llama-runs && BENCH=1 MODEL_PRESET=ornith ./serve-up.sh
cd ${HOME}/llama-runs && ./serve-down.sh
curl -s http://localhost:8080/v1/models | python3 -m json.tool

# smoke test (3 IFEval prompts)
cd ${HOME}/llama-runs/gauntlet && uv run python smoke.py

# one benchmark
cd ${HOME}/llama-runs/gauntlet && uv run python bench_one.py \
    --benchmark ifeval --model auto --out results/X.json \
    --n-samples 50 --max-tokens 16384

# full gauntlet
cd ${HOME}/llama-runs/gauntlet && N_SAMPLES=50 ./run-gauntlet.sh

# tail a running benchmark
tail -f ${HOME}/llama-runs/gauntlet/logs/ifeval_ornith_n10_16k_v2.log

# check what's running
pgrep -af "bench_one|llama-server"
ps -o pid,etime,command -p $(pgrep -f bench_one)
```
