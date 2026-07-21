#!/usr/bin/env bash
# serve.sh — start llama-server with benchmarking flags.
#
# Reference serving configuration used in the LLM Gauntlet benchmarks.
# Not a drop-in replacement for the author's personal serve-up.sh (which
# includes machine-specific Loki log-pushing and fileserver integration).
# Adapt paths to your setup.
#
#   qwen4        unsloth/Qwen3.6-27B-MTP-GGUF / Qwen3.6-27B-UD-Q4_K_XL.gguf
#   qwen6        unsloth/Qwen3.6-27B-MTP-GGUF / Qwen3.6-27B-UD-Q6_K_XL.gguf
#   thinkingcap  bottlecapai/ThinkingCap-Qwen3.6-27B-GGUF / ThinkingCap-Qwen3.6-27B-Q4_K_M.gguf
#   qwen122b     unsloth/Qwen3.5-122B-A10B-MTP-GGUF / Qwen3.5-122B-A10B-UD-Q2_K_XL.gguf
#   ornith       skinnyctax/Ornith-1.0-35B-Q6_K-Frankenstein-MTP-GGUF / ornith-1.0-35b-q6_k.gguf
#   qwythos-v2   empero-ai/Qwythos-9B-v2-GGUF / Qwythos-9B-v2-Q4_K_M.gguf (bundled template, no froggeric)
#   minicpm5-v2  GnLOLot/MiniCPM5-1B-Claude-Opus-Fable5-V2-Thinking-GGUF / MiniCPM5-1B-Claude-Opus-Fable5-V2-Thinking-Q8_0.gguf
#   fable-fusion-mtp  DavidAU/...-NEO-MAX-MTP-GGUF / ...-NEO-MTP-Q4_K_M.gguf
#   custom       set MODEL_REPO + MODEL_FILE in the environment
#
# Env vars:
#   MODEL_PRESET               default: qwen4
#   BENCH=1                    forces N_PARALLEL=1 for reproducible runs
#   USE_FROGGERIC_CHAT_TEMPLATE  1=use froggeric/Qwen-Fixed-Chat-Templates v21.3
#   CHAT_TEMPLATE              path to chat_template.jinja (default: ./chat_template.jinja)
#   HOST / PORT                default: 0.0.0.0 / 8080
#   CTX                        default: 262144
#   LLAMA_BIN                  default: llama-server
#
# The benchmark harness used these exact llama-server flags:
#   --metrics --slots --log-timestamps -t 8 -fa on -c 262144 -np 1
#   --kv-unified --cache-type-k q8_0 --cache-type-v q8_0
#   --spec-type draft-mtp --mlock
#   --jinja --chat-template-file <path>  (when USE_FROGGERIC_CHAT_TEMPLATE=1)
#
# Usage:
#   BENCH=1 MODEL_PRESET=qwen4 USE_FROGGERIC_CHAT_TEMPLATE=1 ./scripts/serve.sh
set -euo pipefail

MODEL_PRESET="${MODEL_PRESET:-qwen4}"
USE_MTP=1  # default: enable MTP speculative decoding
case "$MODEL_PRESET" in
  qwen4)
    MODEL_REPO="unsloth/Qwen3.6-27B-MTP-GGUF"
    MODEL_FILE="Qwen3.6-27B-UD-Q4_K_XL.gguf"
    USE_MTP=0
    ;;
  qwen6)
    MODEL_REPO="unsloth/Qwen3.6-27B-MTP-GGUF"
    MODEL_FILE="Qwen3.6-27B-UD-Q6_K_XL.gguf"
    USE_MTP=0
    ;;
  thinkingcap)
    MODEL_REPO="bottlecapai/ThinkingCap-Qwen3.6-27B-GGUF"
    MODEL_FILE="ThinkingCap-Qwen3.6-27B-Q4_K_M.gguf"
    USE_MTP=0
    ;;
  qwen122b)
    MODEL_REPO="unsloth/Qwen3.5-122B-A10B-MTP-GGUF"
    MODEL_FILE="Qwen3.5-122B-A10B-UD-Q2_K_XL.gguf"
    ;;
  ornith)
    MODEL_REPO="skinnyctax/Ornith-1.0-35B-Q6_K-Frankenstein-MTP-GGUF"
    MODEL_FILE="ornith-1.0-35b-q6_k.gguf"
    ;;
  qwythos-v2)
    MODEL_REPO="empero-ai/Qwythos-9B-v2-GGUF"
    MODEL_FILE="Qwythos-9B-v2-Q4_K_M.gguf"
    USE_MTP=0
    ;;
  minicpm5-v2)
    MODEL_REPO="GnLOLot/MiniCPM5-1B-Claude-Opus-Fable5-V2-Thinking-GGUF"
    MODEL_FILE="MiniCPM5-1B-Claude-Opus-Fable5-V2-Thinking-Q8_0.gguf"
    USE_MTP=0
    ;;
  fable-fusion-mtp)
    # DavidAU Fable-Fusion 27B with native MTP heads (Q4_K_M ~17.2 GB)
    MODEL_REPO="DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF"
    MODEL_FILE="Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q4_K_M.gguf"
    USE_MTP=1
    ;;
  custom)
    : "${MODEL_REPO:?custom preset requires MODEL_REPO env var}"
    : "${MODEL_FILE:?custom preset requires MODEL_FILE env var}"
    ;;
  *)
    echo "ERROR: unknown MODEL_PRESET: $MODEL_PRESET (use: qwen4|qwen6|thinkingcap|qwen122b|ornith|qwythos-v2|minicpm5-v2|fable-fusion-mtp|custom)" >&2
    exit 1
    ;;
esac

LLAMA_BIN="${LLAMA_BIN:-llama-server}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
CTX="${CTX:-262144}"
N_PARALLEL="${N_PARALLEL:-4}"

# BENCH=1 forces single-slot mode for reproducible benchmarking
if [ "${BENCH:-0}" = "1" ]; then
    N_PARALLEL=1
fi

# froggeric chat template
USE_FROGGERIC_CHAT_TEMPLATE="${USE_FROGGERIC_CHAT_TEMPLATE:-0}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$(dirname "$0")/../chat_template.jinja}"
EXTRA_ARGS=""
if [ "$USE_FROGGERIC_CHAT_TEMPLATE" = "1" ]; then
    if [ -f "$CHAT_TEMPLATE" ]; then
        EXTRA_ARGS="--jinja --chat-template-file $CHAT_TEMPLATE"
        echo "[serve] using chat template: $CHAT_TEMPLATE (froggeric/Qwen-Fixed-Chat-Templates v21.3)"
    else
        echo "WARNING: chat template not found at $CHAT_TEMPLATE" >&2
    fi
fi

command -v "$LLAMA_BIN" >/dev/null || { echo "ERROR: $LLAMA_BIN not in PATH (brew install llama.cpp)" >&2; exit 1; }

# Kill any existing server on the same port
existing=$(pgrep -f "$LLAMA_BIN.*--port $PORT" || true)
if [ -n "$existing" ]; then
    echo "[serve] killing existing llama-server on port $PORT (pid $existing)"
    kill $existing 2>/dev/null || true
    sleep 2
fi

echo "[serve] starting llama-server (preset=$MODEL_PRESET, port=$PORT, ctx=$CTX, np=$N_PARALLEL, threads=8, mtp=$USE_MTP)"
# Build argv as an array so multi-token flags like --spec-type draft-mtp stay split.
CMD=(
    "$LLAMA_BIN"
    --hf-repo "$MODEL_REPO"
    --hf-file "$MODEL_FILE"
    --host "$HOST"
    --port "$PORT"
    --metrics
    --slots
    --log-timestamps
    -t 8
    -fa on
    -c "$CTX"
    -np "$N_PARALLEL"
    --kv-unified
    --cache-type-k q8_0
    --cache-type-v q8_0
    --mlock
)
if [ "$USE_MTP" = "1" ]; then
    CMD+=(--spec-type draft-mtp --spec-draft-n-max 2)
fi
if [ -n "$EXTRA_ARGS" ]; then
    # shellcheck disable=SC2206
    CMD+=($EXTRA_ARGS)
fi
nohup "${CMD[@]}" > /tmp/llama-server.stdout 2>&1 &
server_pid=$!
echo "$server_pid" > /tmp/llama-server.pid

# Wait for /health (first HF download can take longer than 180s — keep polling).
echo "[serve] waiting for /health (timeout 900s)..."
ready=0
for _ in $(seq 1 900); do
    if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
        || curl -fsS --max-time 2 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    # Bail early if the server process already exited.
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "ERROR: llama-server exited during startup — see /tmp/llama-server.stdout" >&2
        tail -n 40 /tmp/llama-server.stdout >&2 || true
        exit 1
    fi
    sleep 1
done
if [ "$ready" != "1" ]; then
    echo "ERROR: llama-server did not become healthy within 900s" >&2
    tail -n 40 /tmp/llama-server.stdout >&2 || true
    exit 1
fi

echo ""
echo "llama-server up."
echo "  model:  $MODEL_REPO / $MODEL_FILE"
echo "  url:    http://$HOST:$PORT/v1"
echo "  pid:    $server_pid"
echo ""
echo "Check: curl -s http://$HOST:$PORT/v1/models | python3 -m json.tool"
echo "Kill:  kill \$(cat /tmp/llama-server.pid)"
