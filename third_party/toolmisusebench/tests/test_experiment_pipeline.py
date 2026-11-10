from __future__ import annotations

import json

from toolmisusebench.cli.main import main
from toolmisusebench.experiment_pipeline import load_experiment_config, run_experiment_pipeline


def test_run_all_writes_results_tables_and_plots(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    dataset_path = tmp_path / "data" / "v0_1"
    out_path = tmp_path / "out" / "experiments" / "v0_1"
    config_path.write_text(
        "\n".join(
            [
                "version: v0.1",
                "seed: 7",
                f"dataset: {dataset_path}",
                "split: test_public",
                "agents:",
                "  - heuristic",
                "  - schema_repair",
                "  - policy_aware",
                f"out_dir: {out_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)
    payload = run_experiment_pipeline(config)

    assert (out_path / "results.json").exists()
    assert (out_path / "table1_overall.csv").exists()
    assert (out_path / "table2_fault_breakdown.csv").exists()
    assert (out_path / "figure1_budgeted_success.csv").exists()
    assert (out_path / "figure2_time_to_recovery.csv").exists()
    assert payload["report"].endswith("results.json")

    results = json.loads((out_path / "results.json").read_text(encoding="utf-8"))
    assert set(results["by_agent"].keys()) == {"heuristic", "schema_repair", "policy_aware"}


def test_cli_reproduce_paper_runs_experiment_pipeline(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "version: v0.1",
                "seed: 11",
                f"dataset: {tmp_path / 'dataset'}",
                "split: test_public",
                "agents: [heuristic]",
                f"out_dir: {tmp_path / 'paper_out'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(["reproduce-paper", "--config", str(config_path)])
    assert exit_code == 0
    assert (tmp_path / "paper_out" / "results.json").exists()
