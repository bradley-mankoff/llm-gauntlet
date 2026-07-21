#!/usr/bin/env bash
# Same locked scout recipe as the MTPLX 27B graphify run, but served via
# llama.cpp with DavidAU Fable-Fusion 27B NEO-MTP Q4_K_M.
#
# Recipe (match pipeline_27b_mtplx_graphify.json):
#   embedder: Qwen3-Embedding-0.6B
#   graphify: ON, uncapped novel-file merge, BFS depth 2
#   reranker: BGE-Reranker-v2-m3
#   top_n: 3
#   sleep: 10s between queries
#   think-off via froggeric hard switch
#   checkpoint + --resume
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
REPO="${REPO:-/opt/homebrew/var/mtplx/venv-2.0.2/lib/python3.13/site-packages/mtplx}"
QUERIES="${QUERIES:-$ROOT/queries/scout_mtplx.json}"
OUT="${OUT:-$ROOT/results/pipeline_fable_fusion_mtp_q4km_graphify.json}"
TOP_N="${TOP_N:-3}"
SLEEP_S="${SLEEP_S:-10}"
RETRIES="${RETRIES:-5}"
BENCH_LOG="${BENCH_LOG:-$ROOT/logs/pipeline_fable_fusion_mtp_q4km_graphify.log}"
PID_FILE="${PID_FILE:-/tmp/fable-fusion-graphify.pid}"

if [[ ! -d "$REPO" ]]; then
  echo "ERROR: MTPLX package repo not found: $REPO" >&2
  exit 1
fi
if [[ ! -f "$REPO/graphify-out/graph.json" ]]; then
  echo "ERROR: missing $REPO/graphify-out/graph.json" >&2
  exit 1
fi
if [[ ! -f "$QUERIES" ]]; then
  echo "ERROR: missing queries file: $QUERIES" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")" "$(dirname "$BENCH_LOG")"

health_ok() {
  curl -fsS --max-time 5 "http://${HOST}:${PORT}/health" >/dev/null 2>&1 \
    || curl -fsS --max-time 5 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1
}

start_server() {
  if health_ok; then
    echo "[run] server already healthy on ${HOST}:${PORT}"
    curl -s "http://${HOST}:${PORT}/v1/models" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(" model:", d["data"][0]["id"] if d.get("data") else d)' 2>/dev/null || true
    return 0
  fi
  echo "[run] starting llama-server Fable-Fusion MTP Q4_K_M + froggeric think-off"
  mtplx stop --port "$PORT" >/dev/null 2>&1 || true
  if command -v lsof >/dev/null 2>&1; then
    lsof -ti ":${PORT}" | xargs kill -9 2>/dev/null || true
  fi
  sleep 1
  LOCAL_GGUF="${LOCAL_GGUF:-$HOME/llama-runs/models/fable-fusion-mtp-q4km/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q4_K_M.gguf}"
  TEMPLATE="${CHAT_TEMPLATE:-$ROOT/chat_template.jinja}"
  CTX="${CTX:-65536}"
  if [[ -f "$LOCAL_GGUF" ]]; then
    echo "[run] using local GGUF: $LOCAL_GGUF"
    nohup llama-server \
      --model "$LOCAL_GGUF" \
      --host "$HOST" --port "$PORT" \
      --metrics --slots --log-timestamps \
      -t 8 -fa on -c "$CTX" -np 1 \
      --kv-unified --cache-type-k q8_0 --cache-type-v q8_0 \
      --spec-type draft-mtp --spec-draft-n-max 2 \
      --mlock \
      --jinja --chat-template-file "$TEMPLATE" \
      > /tmp/llama-server.stdout 2>&1 &
    echo $! > /tmp/llama-server.pid
  else
    echo "[run] local GGUF missing; falling back to HF download via serve.sh"
    BENCH=1 MODEL_PRESET=fable-fusion-mtp USE_FROGGERIC_CHAT_TEMPLATE=1 \
      HOST="$HOST" PORT="$PORT" CTX="$CTX" "$ROOT/scripts/serve.sh"
  fi
  for i in $(seq 1 300); do
    if health_ok; then
      echo "[run] server ready after ${i}s"
      return 0
    fi
    if [[ -f /tmp/llama-server.pid ]] && ! kill -0 "$(cat /tmp/llama-server.pid)" 2>/dev/null; then
      echo "ERROR: llama-server died — see /tmp/llama-server.stdout" >&2
      tail -n 40 /tmp/llama-server.stdout >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "ERROR: server not healthy within 300s" >&2
  return 1
}

start_server

RESUME_ARGS=""
if [[ -f "$OUT" ]]; then
  RESUME_ARGS="--resume"
  echo "[run] found existing $OUT — enabling --resume"
fi

echo "[run] launching detached bench -> $OUT"
echo "[run] bench log: $BENCH_LOG"
nohup uv run python bench_pipeline.py \
    --model auto \
    --base-url "http://${HOST}:${PORT}/v1" \
    --repo "$REPO" \
    --queries "$QUERIES" \
    --top-n "$TOP_N" \
    --sleep "$SLEEP_S" \
    --retries "$RETRIES" \
    --think-off \
    --out "$OUT" \
    $RESUME_ARGS \
    >"$BENCH_LOG" 2>&1 &
BENCH_PID=$!
echo "$BENCH_PID" >"$PID_FILE"
echo "[run] bench pid=$BENCH_PID"
echo "[run] tail -f $BENCH_LOG"
echo "[run] summary when done:"
echo "  python -c \"import json;print(json.load(open('$OUT'))['summary'])\""
