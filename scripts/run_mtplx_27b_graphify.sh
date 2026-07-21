#!/usr/bin/env bash
# Reproduce the MTPLX Qwen3.6-27B + graphify BFS scout recipe that hit
# 16/16 file-perfect before the previous run died / timed out.
#
# Locked settings (from the 2026-07-20 partial run):
#   server:  mtplx quickstart --profile stable --max-idle-min 0 --reasoning off
#   model:   Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed
#   scout:   embedder (Qwen3-Embedding-0.6B) + uncapped graphify BFS merge
#            + BGE-Reranker-v2-m3 + 27B LLM, top_n=3
#   harness: 10s drain sleep, per-query checkpoint, --resume, chat retries
#
# Runs DETACHED so an agent/tool 1h timeout cannot kill the client mid-bench.
# Resume is automatic if --out already has partial results.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
MODEL_HF="${MODEL_HF:-Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed}"
REPO="${REPO:-/opt/homebrew/var/mtplx/venv-2.0.2/lib/python3.13/site-packages/mtplx}"
QUERIES="${QUERIES:-$ROOT/queries/scout_mtplx.json}"
OUT="${OUT:-$ROOT/results/pipeline_27b_mtplx_graphify.json}"
TOP_N="${TOP_N:-3}"
SLEEP_S="${SLEEP_S:-10}"
RETRIES="${RETRIES:-5}"
SERVER_LOG="${SERVER_LOG:-/tmp/mtplx-27b-graphify.log}"
BENCH_LOG="${BENCH_LOG:-$ROOT/logs/pipeline_27b_mtplx_graphify.log}"
PID_FILE="${PID_FILE:-/tmp/mtplx-27b-graphify.pid}"

if [[ ! -d "$REPO" ]]; then
  echo "ERROR: MTPLX package repo not found: $REPO" >&2
  exit 1
fi
if [[ ! -f "$REPO/graphify-out/graph.json" ]]; then
  echo "ERROR: missing $REPO/graphify-out/graph.json — run graphify on the MTPLX package first" >&2
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
    echo "[run] MTPLX already healthy on ${HOST}:${PORT}"
    return 0
  fi
  echo "[run] starting MTPLX (stable, max-idle-min=0) on :${PORT}"
  # Free the port if a dead listener remains.
  if command -v lsof >/dev/null 2>&1; then
    lsof -ti ":${PORT}" | xargs kill -9 2>/dev/null || true
  fi
  # Prefer stop via mtplx if available.
  mtplx stop --port "$PORT" >/dev/null 2>&1 || true
  sleep 2
  nohup mtplx quickstart \
      --model "$MODEL_HF" \
      --port "$PORT" --host "$HOST" \
      --reasoning off \
      --profile stable \
      --max-idle-min 0 \
      >"$SERVER_LOG" 2>&1 &
  echo $! >"$PID_FILE"
  # Wait for health (model load ~minutes on cold start, seconds when cached).
  for i in $(seq 1 180); do
    if health_ok; then
      echo "[run] MTPLX ready after ${i}s (log: $SERVER_LOG)"
      return 0
    fi
    sleep 2
  done
  echo "ERROR: MTPLX failed to become healthy within 360s — see $SERVER_LOG" >&2
  tail -n 80 "$SERVER_LOG" >&2 || true
  return 1
}

start_server

RESUME_FLAG=()
if [[ -f "$OUT" ]]; then
  # Auto-resume when partial/complete output exists.
  RESUME_FLAG=(--resume)
  echo "[run] found existing $OUT — enabling --resume"
fi

echo "[run] launching detached bench -> $OUT"
echo "[run] bench log: $BENCH_LOG"
# Detached client: survives shell exit and agent tool timeouts.
nohup uv run python bench_pipeline.py \
    --model auto \
    --base-url "http://${HOST}:${PORT}/v1" \
    --repo "$REPO" \
    --queries "$QUERIES" \
    --top-n "$TOP_N" \
    --sleep "$SLEEP_S" \
    --retries "$RETRIES" \
    --out "$OUT" \
    "${RESUME_FLAG[@]}" \
    >"$BENCH_LOG" 2>&1 &
BENCH_PID=$!
echo "$BENCH_PID" >"${PID_FILE}.bench"
echo "[run] bench pid=$BENCH_PID"
echo "[run] tail -f $BENCH_LOG"
echo "[run] when finished: python -c \"import json;print(json.load(open('$OUT'))['summary'])\""
