"""Build a clean Hugging Face dataset export folder for ToolMisuseBench."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data/toolmisusebench_v0_2_large"
DEFAULT_RUN = ROOT / "out/release_runs/v0_2_large"


README = """---
license: mit
task_categories:
- text-generation
pretty_name: ToolMisuseBench
configs:
- config_name: v0_2_large
  data_files:
  - split: train
    path: data/v0_2_large/train.jsonl
  - split: validation
    path: data/v0_2_large/dev.jsonl
  - split: test
    path: data/v0_2_large/test_public.jsonl
---

# ToolMisuseBench

ToolMisuseBench is an offline deterministic benchmark for evaluating tool misuse and recovery in agentic systems. It provides synthetic but controlled tool-use tasks with replayable fault plans, typed recoverability metadata, output-grounded success criteria, deterministic reference agents, and scored experiment artifacts.

## Current Release

This export contains the seed-42 `v0.2` large release used for the current ToolMisuseBench paper draft:

- train: 6,000 tasks
- validation/dev: 900 tasks
- public test: 1,200 tasks
- domains: CRUD, retrieval, files, scheduling
- fault families: clean, timeout, rate limit, schema drift, authorization, adversarial error
- difficulty labels: easy, medium, hard
- budget profiles: tight, standard, relaxed

The public test split contains 600 output-grounded tasks and no duplicate task identifiers. Release diagnostics are included under `artifacts/v0_2_large/`.

## Files

- `data/v0_2_large/train.jsonl`
- `data/v0_2_large/dev.jsonl`
- `data/v0_2_large/test_public.jsonl`
- `artifacts/v0_2_large/manifest.json`
- `artifacts/v0_2_large/v0_2_freeze.json`
- `artifacts/v0_2_large/dataset_quality_report.json`
- `artifacts/v0_2_large/release_summary.json`
- `artifacts/v0_2_large/fault_ablation_summary.json`
- `artifacts/v0_2_large/budget_sweep_deterministic.json`
- `artifacts/v0_2_large/budget_sweep_llm.json`
- `artifacts/v0_2_large/stages/*/score.json`
- `artifacts/v0_2_large/stages/*/metadata.json`
- `artifacts/v0_2_large/stages/*/episodes.jsonl`

## Intended Use

ToolMisuseBench is intended for research on:

- tool-call argument correctness
- recovery after transient tool failures
- schema drift and interface mismatch
- authorization-denial behavior
- ambiguous or adversarial tool feedback
- fault-conditioned evaluation of tool-using agents

It is not intended as a broad web-agent benchmark, live API reliability benchmark, cybersecurity benchmark, or substitute for production integration testing.

## Reproducibility

The benchmark is deterministic under fixed seeds. Each release includes a manifest with SHA-256 checksums and freeze metadata. The corresponding code repository contains the generator, evaluator, reference agents, scoring pipeline, and experiment scripts:

https://github.com/akgitrepos/toolmisusebench

## Citation

If you use this dataset, please cite:

```bibtex
@misc{sigdel2026toolmisusebench,
  title = {ToolMisuseBench: Deterministic Evaluation of Tool Misuse, Typed Recoverability, and Fault-Conditioned Recovery},
  author = {Sigdel, Akshey and Baral, Rista},
  year = {2026},
  howpublished = {Hugging Face Datasets},
  url = {https://huggingface.co/datasets/sigdelakshey/ToolMisuseBench}
}
```
"""


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_export(out_dir: Path, dataset_dir: Path = DEFAULT_DATASET, run_dir: Path = DEFAULT_RUN) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    (out_dir / "README.md").write_text(README, encoding="utf-8")

    copy_file(dataset_dir / "train/tasks.jsonl", out_dir / "data/v0_2_large/train.jsonl")
    copy_file(dataset_dir / "dev/tasks.jsonl", out_dir / "data/v0_2_large/dev.jsonl")
    copy_file(dataset_dir / "test_public/tasks.jsonl", out_dir / "data/v0_2_large/test_public.jsonl")
    copy_file(dataset_dir / "manifest.json", out_dir / "artifacts/v0_2_large/manifest.json")
    copy_file(dataset_dir / "v0_2_freeze.json", out_dir / "artifacts/v0_2_large/v0_2_freeze.json")

    for name in [
        "dataset_quality_report.json",
        "release_summary.json",
        "fault_ablation_summary.json",
        "budget_sweep_deterministic.json",
        "budget_sweep_llm.json",
        "run_manifest.json",
        "run_status.json",
    ]:
        copy_file(run_dir / name, out_dir / f"artifacts/v0_2_large/{name}")

    stages_out = out_dir / "artifacts/v0_2_large/stages"
    for stage_dir in sorted((run_dir / "stages").iterdir()):
        if not stage_dir.is_dir():
            continue
        copy_file(stage_dir / "score.json", stages_out / stage_dir.name / "score.json")
        copy_file(stage_dir / "metadata.json", stages_out / stage_dir.name / "metadata.json")
        copy_file(stage_dir / "episodes.jsonl", stages_out / stage_dir.name / "episodes.jsonl")

    summary = {
        "dataset_dir": str(dataset_dir.relative_to(ROOT)),
        "run_dir": str(run_dir.relative_to(ROOT)),
        "files": sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file()),
    }
    (out_dir / "EXPORT_MANIFEST.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Clean export directory to create.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    build_export(args.out, args.dataset, args.run)
    print(args.out)


if __name__ == "__main__":
    main()
