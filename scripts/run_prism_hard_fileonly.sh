#!/usr/bin/env bash
# Harder file-only scout eval on prism-llama.cpp (multi-lang, no-symbol queries).
# Uses local Fable-Fusion MTP Q4_K_M with candidate-locked file-only prompt.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
REPO="${REPO:-$ROOT/../prism-llama.cpp}"
QUERIES="${QUERIES:-$ROOT/queries/scout_prism_hard_file.json}"
OUT="${OUT:-$ROOT/results/pipeline_fable_prism_hard_fileonly.json}"
TOP_N="${TOP_N:-5}"
SLEEP_S="${SLEEP_S:-5}"
BENCH_LOG="${BENCH_LOG:-$ROOT/logs/pipeline_fable_prism_hard_fileonly.log}"
LOCAL_GGUF="${LOCAL_GGUF:-$HOME/llama-runs/models/fable-fusion-mtp-q4km/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q4_K_M.gguf}"
TEMPLATE="${CHAT_TEMPLATE:-$ROOT/chat_template.jinja}"

if [[ ! -d "$REPO" ]]; then
  echo "ERROR: repo not found: $REPO" >&2
  exit 1
fi
if [[ ! -f "$QUERIES" ]]; then
  echo "ERROR: queries not found: $QUERIES" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")" "$(dirname "$BENCH_LOG")"

health_ok() {
  curl -fsS --max-time 5 "http://${HOST}:${PORT}/health" >/dev/null 2>&1
}

if ! health_ok; then
  echo "[run] starting Fable-Fusion server on :$PORT"
  mtplx stop --port "$PORT" >/dev/null 2>&1 || true
  lsof -ti ":$PORT" | xargs kill -9 2>/dev/null || true
  sleep 1
  nohup llama-server \
    --model "$LOCAL_GGUF" \
    --host "$HOST" --port "$PORT" \
    -t 8 -fa on -c "${CTX:-65536}" -np 1 \
    --kv-unified --cache-type-k q8_0 --cache-type-v q8_0 \
    --spec-type draft-mtp --spec-draft-n-max 2 \
    --jinja --chat-template-file "$TEMPLATE" \
    >/tmp/llama-server.stdout 2>&1 &
  echo $! >/tmp/llama-server.pid
  for i in $(seq 1 120); do
    health_ok && break
    sleep 1
  done
  health_ok || { echo "server failed"; tail -40 /tmp/llama-server.stdout; exit 1; }
fi

# Ensure index exists for prism (C/C++ + py)
if [[ ! -d "$REPO/.scout_index" ]]; then
  echo "[run] building scout index for $REPO (first time)..."
  uv run python - <<PY
from scout_pipeline import ScoutPipeline
p = ScoutPipeline("$REPO")
p.index(["**/*.py", "**/*.c", "**/*.cc", "**/*.cpp", "**/*.h", "**/*.hpp"], force=True)
print("index done")
PY
fi

RESUME=""
[[ -f "$OUT" ]] && RESUME="--resume"

echo "[run] detached file-only hard eval -> $OUT"
nohup uv run python bench_pipeline.py \
  --model auto \
  --base-url "http://${HOST}:${PORT}/v1" \
  --repo "$REPO" \
  --queries "$QUERIES" \
  --top-n "$TOP_N" \
  --sleep "$SLEEP_S" \
  --retries 4 \
  --think-off \
  --file-only \
  --no-graphify \
  --max-tokens 64 \
  --out "$OUT" \
  $RESUME \
  >"$BENCH_LOG" 2>&1 &
echo "bench_pid=$! log=$BENCH_LOG"
echo "tail -f $BENCH_LOG"
