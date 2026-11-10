#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-out/release_runs/v0_2_large}"
DATASET_DIR="${DATASET_DIR:-data/toolmisusebench_v0_2_large}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8080/v1}"
LLM_MODEL="${LLM_MODEL:-Qwen3-4B-Q4_K_M}"
LABEL="${LABEL:-org.toolmisusebench.v2release}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$PROJECT_DIR/$OUT_DIR"
launchctl bootout "gui/$UID/$LABEL" 2>/dev/null || true
launchctl submit -l "$LABEL" \
  -o "$PROJECT_DIR/$OUT_DIR/release.stdout.log" \
  -e "$PROJECT_DIR/$OUT_DIR/release.stderr.log" \
  -- /bin/bash -lc "cd \"$PROJECT_DIR\" && exec \"$PROJECT_DIR/.venv/bin/python\" -m toolmisusebench.cli.main run-v2-release --out \"$OUT_DIR\" --dataset \"$DATASET_DIR\" --llm-base-url \"$LLM_BASE_URL\" --llm-model \"$LLM_MODEL\" --calibration-tasks 48 --llm-panel-tasks 216 --llm-sweep-tasks-per-cap 48 --llm-budget-caps 2,4,6,10 --llm-max-tokens 128 --llm-progress-every 5 --max-llm-minutes 180"

echo "$LABEL" > "$PROJECT_DIR/$OUT_DIR/launchctl-label"
echo "Started release run as launchd job $LABEL."
echo "Status: $PROJECT_DIR/.venv/bin/python -m toolmisusebench.cli.main status-v2-release --out $OUT_DIR"
