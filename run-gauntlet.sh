#!/usr/bin/env bash
# Run all 3 benchmarks against all 5 model presets. Sequential, hands-free.
#
# For each preset:
#   1. serve-up with BENCH=1 (single slot, reproducible). Qwen-based presets
#      also get USE_FROGGERIC_CHAT_TEMPLATE=1 to swap in the
#      froggeric/Qwen-Fixed-Chat-Templates Jinja template (fixes the upstream
#      Qwen3.6 chat-template bugs that cause infinite thinking + KV cache
#      thrash). Ornith and ThinkingCap both leave it at 0.
#   2. run 3 benchmarks back-to-back against the live server
#   3. serve-down
# Total: 5 models × 3 benchmarks = 15 JSON files in ./results/
#
# Knobs:
#   N_SAMPLES=50  ./run-gauntlet.sh   # quick smoke run, 15 * 50 = 750 evals
#   N_SAMPLES=    ./run-gauntlet.sh   # full run, 15 * 541 = 8115 evals
#   SKIP_DOWN=1   ./run-gauntlet.sh   # leave server up between presets
set -euo pipefail
cd "$(dirname "$0")"

PRESETS=(qwen4 thinkingcap)
BENCHMARKS=(ifeval humaneval repoqa)
N_SAMPLES="${N_SAMPLES:-}"   # empty = full benchmark
MAX_TOKENS_IFEVAL="${MAX_TOKENS_IFEVAL:-8192}"
BENCHMARKS=(ifeval humaneval repoqa)
mkdir -p results logs

for preset in "${PRESETS[@]}"; do
    echo "================================================================"
    echo "[gauntlet] serving preset=$preset (BENCH=1, single slot)"
    echo "================================================================"
    # Qwen-based presets (qwen4/5/6 and the Qwen-architecture ThinkingCap) get the
    # froggeric chat template. Ornith is a Frankenstein merge with a different
    # token format — its bundled template already produces clean IFEval results.
    case "$preset" in
        ornith) USE_FROGGERIC=0 ;;
        *)       USE_FROGGERIC=1 ;;
    esac
    ( cd "$LLAMA_DIR" && BENCH=1 MODEL_PRESET="$preset" USE_FROGGERIC_CHAT_TEMPLATE=$USE_FROGGERIC ./serve-up.sh )
    sleep 2

    # Detect the model name the server is now serving
    model=$(curl -s "$BASE_URL/v1/models" | python3 -c "import sys, json; d = json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "")
    if [ -z "$model" ]; then
        echo "[gauntlet] ERROR: could not read served model from $BASE_URL/v1/models" >&2
        if [ "${SKIP_DOWN:-0}" != "1" ]; then
            ( cd "$LLAMA_DIR" && ./serve-down.sh ) || true
        fi
        continue
    fi
    safe_model=$(echo "$model" | tr '/: ' '___')

    for bench in "${BENCHMARKS[@]}"; do
        echo "----------------------------------------------------------------"
        echo "[gauntlet] preset=$preset benchmark=$bench"
        echo "----------------------------------------------------------------"
        extra_args=()
        if [ -n "$N_SAMPLES" ]; then
            extra_args+=(--n-samples "$N_SAMPLES")
        fi
        if [ "$bench" = "ifeval" ] && [ -n "$MAX_TOKENS_IFEVAL" ]; then
            extra_args+=(--max-tokens "$MAX_TOKENS_IFEVAL")
        fi
        # CodeEditBench and RepoQA are currently NotImplementedError scaffolds;
        # capture that and move on so the loop doesn't die.
        if uv run python bench_one.py \
            --benchmark "$bench" \
            --model "$model" \
            --base-url "$BASE_URL" \
            --out "results/${preset}__${bench}.json" \
            "${extra_args[@]}" 2>&1 | tee "logs/${preset}__${bench}.log"; then
            :
        else
            echo "[gauntlet] benchmark $bench on $preset failed (or is a scaffold) — continuing"
        fi
    done

    echo "[gauntlet] tearing down server"
    if [ "${SKIP_DOWN:-0}" != "1" ]; then
        ( cd "$LLAMA_DIR" && ./serve-down.sh ) || true
    fi
    sleep 5
done

echo "================================================================"
echo "[gauntlet] all runs complete. Results in ./results/"
echo "================================================================"
ls -la results/
