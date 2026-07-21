#!/usr/bin/env bash
# Restart llama-server + resume bench_pipeline until status=complete or max attempts.
# macOS /bin/bash 3.2 compatible (no mapfile).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:?OUT required}"
LOG="${LOG:?LOG required}"
REPO="${REPO:?REPO required}"
QUERIES="${QUERIES:?QUERIES required}"
PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-40}"
LOCAL_GGUF="${LOCAL_GGUF:-$HOME/llama-runs/models/fable-fusion-mtp-q4km/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q4_K_M.gguf}"
TEMPLATE="${CHAT_TEMPLATE:-$ROOT/chat_template.jinja}"
EXTRA_ARGS="${EXTRA_ARGS:---top-n 5 --sleep 5 --retries 4 --think-off --file-only --no-graphify --max-tokens 64}"

health() { curl -fsS --max-time 3 "http://${HOST}:${PORT}/health" >/dev/null 2>&1; }

ensure_server() {
  if health; then return 0; fi
  echo "[sup] starting server"
  pkill -9 -f llama-server 2>/dev/null || true
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
  for _ in $(seq 1 120); do health && return 0; sleep 1; done
  echo "[sup] server failed" >&2
  tail -30 /tmp/llama-server.stdout >&2 || true
  return 1
}

read_status() {
  # sets: CUR_STATUS N_DONE N_TARGET FILE1 RET3
  eval "$(python3 - <<PY
import json
from pathlib import Path
p=Path(r'''$OUT''')
if not p.exists():
    print('CUR_STATUS=missing')
    print('N_DONE=0')
    print('N_TARGET=0')
    print('FILE1=')
    print('RET3=')
    raise SystemExit
d=json.loads(p.read_text())
s=d.get('summary',{})
def esc(x):
    return str(x).replace("'", "")
print('CUR_STATUS=%s' % esc(s.get('status') or 'partial'))
print('N_DONE=%s' % esc(s.get('n_samples') or 0))
print('N_TARGET=%s' % esc(s.get('n_target') or 0))
print('FILE1=%s' % esc(s.get('file_match_rate')))
print('RET3=%s' % esc(s.get('file_at_3')))
PY
)"
}

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  read_status
  echo "[sup] attempt=$attempt status=$CUR_STATUS n=$N_DONE/$N_TARGET file@1=$FILE1 ret@3=$RET3"
  if [ "$CUR_STATUS" = complete ]; then
    echo "[sup] done"
    exit 0
  fi
  ensure_server || continue
  pkill -f 'bench_pipeline.py' 2>/dev/null || true
  sleep 1
  RESUME=""
  if [ -f "$OUT" ]; then RESUME="--resume"; fi
  # shellcheck disable=SC2086
  uv run python bench_pipeline.py \
    --model auto --base-url "http://${HOST}:${PORT}/v1" \
    --repo "$REPO" \
    --queries "$QUERIES" \
    $EXTRA_ARGS \
    --out "$OUT" \
    $RESUME \
    >>"$LOG" 2>&1 || true
  sleep 2
done
echo "[sup] exhausted attempts" >&2
read_status
echo "[sup] final status=$CUR_STATUS n=$N_DONE/$N_TARGET"
exit 1
