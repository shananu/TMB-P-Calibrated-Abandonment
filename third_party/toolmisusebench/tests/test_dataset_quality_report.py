from __future__ import annotations

from toolmisusebench.generator import analyze_dataset, generate_dataset


def test_analyze_dataset_reports_expected_summary(tmp_path) -> None:
    out = tmp_path / "data"
    generate_dataset(
        version="v0.1",
        out_dir=out,
        seed=2,
        write_parquet=False,
        split_sizes={"train": 40, "dev": 10, "test_public": 10},
    )

    summary = analyze_dataset(out)
    assert summary["overall"]["rows"] == 60
    assert set(summary["splits"]) == {"train", "dev", "test_public"}
    assert summary["per_split"]["train"]["rows"] == 40
    assert summary["per_split"]["train"]["duplicate_task_ids"] == 0
