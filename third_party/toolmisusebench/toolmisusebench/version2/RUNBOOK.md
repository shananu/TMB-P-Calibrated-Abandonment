# ToolMisuseBench v2 Release Runbook

The release runner records every experiment stage, per-task trace, score payload, runtime estimate, dataset-quality report, configuration, and event in an artifact directory.

## Stages

1. Generate or reuse the v2 large dataset.
2. Run all deterministic baselines on the full 1,200-task public test split.
3. Run a balanced LLM calibration panel of 48 tasks.
4. Estimate wall-clock time for the 216-task LLM panel.
5. Run the LLM balanced panel plus small enforced-cap sweep panels only when the estimate is at or below the configured runtime limit, unless forced.

The LLM panel contains a balanced sample across domain, primary fault, difficulty, and budget profile. Responses are capped at 128 tokens and reasoning is disabled by default because every benchmark step requires only one structured tool call. Use `--llm-thinking` only for a separate sensitivity run.

## Local Model Server

The current laptop is best matched to Qwen3-4B-GGUF at Q4 quantization. Start an OpenAI-compatible local server before starting the release runner. With a compatible `llama.cpp` installation, the shape is:

```bash
llama-server -hf Qwen/Qwen3-4B-GGUF:Q4_K_M -ngl 99 -c 4096 --jinja
```

If the server supports a Qwen tool-call parser, enable it according to that server version. The runner also accepts JSON action content as a fallback when native tool calls are not returned.

## Start in the Background

```bash
bash experiments/run_v2_release.sh
```

The background launcher writes `pid` and `console.log` under `out/release_runs/v0_2_large/`.

## Monitor Progress

```bash
.venv/bin/python -m toolmisusebench.cli.main \
  status-v2-release --out out/release_runs/v0_2_large
```

Live files:

- `run_status.json`: current stage, task counts, elapsed time, and ETA
- `run_events.jsonl`: append-only event history
- `run_manifest.json`: exact run configuration and host metadata
- `dataset_quality_report.json`: deconfounding diagnostics
- `stages/<stage>/episodes.jsonl`: per-task traces, safe to resume from
- `stages/<stage>/score.json`: final metrics for a completed stage
- `release_summary.json`: combined completed-stage summary
- `fault_ablation_summary.json`: clean and fault-family performance slices
- `budget_sweep_deterministic.json`: enforced-cap curves for recovery ablations
- `budget_sweep_llm.json`: enforced-cap curve data for the LLM baseline

## Resume Behavior

Running the same command again with the same `--out` directory skips completed stages and resumes an interrupted stage from its trace file. Do not modify trace files manually.
