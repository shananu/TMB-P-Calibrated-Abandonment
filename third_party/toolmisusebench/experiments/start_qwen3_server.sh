#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-out/release_runs/v0_2_large}"
LABEL="${LABEL:-org.toolmisusebench.qwen3server}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$PROJECT_DIR/$OUT_DIR"
launchctl bootout "gui/$UID/$LABEL" 2>/dev/null || true
launchctl submit -l "$LABEL" \
  -o "$PROJECT_DIR/$OUT_DIR/qwen-server.stdout.log" \
  -e "$PROJECT_DIR/$OUT_DIR/qwen-server.stderr.log" \
  -- /opt/homebrew/bin/llama-server \
  -hf Qwen/Qwen3-4B-GGUF:Q4_K_M \
  --jinja \
  -ngl 0 \
  -c 2048 \
  --host 127.0.0.1 \
  --port 8080

echo "$LABEL" > "$PROJECT_DIR/$OUT_DIR/qwen-server-launchctl-label"
echo "Started Qwen server as launchd job $LABEL."
