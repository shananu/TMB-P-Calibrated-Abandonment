from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from toolmisusebench.baselines import build_agent
from toolmisusebench.dataset import load_tasks
from toolmisusebench.generator import generate_dataset
from toolmisusebench.harness.artifacts import write_json
from toolmisusebench.harness.metrics import episode_to_trace_row
from toolmisusebench.harness.runner import run_episode
from toolmisusebench.harness.scoring import score_episodes
from toolmisusebench.version2 import generate_dataset_v2


DEFAULT_CONFIG: dict[str, Any] = {
    "version": "v0.1",
    "seed": 42,
    "dataset": "data/toolmisusebench_v0_1",
    "split": "test_public",
    "agents": ["heuristic", "schema_repair", "policy_aware"],
    "out_dir": "out/experiments/v0_1",
}


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8")
    raw = raw.strip()
    if not raw:
        return dict(DEFAULT_CONFIG)

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return _normalize_config(parsed)
    except json.JSONDecodeError:
        pass

    parsed = _parse_simple_yaml(raw)
    return _normalize_config(parsed)


def run_experiment_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    cfg = _normalize_config(config)
    dataset_path = Path(cfg["dataset"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        if cfg["version"] in {"v0.2", "v2", "version2"}:
            generate_dataset_v2(
                version="v0.2",
                out_dir=dataset_path,
                seed=int(cfg["seed"]),
                size_profile=str(cfg.get("size_profile", "default")),
            )
        else:
            generate_dataset(version=cfg["version"], out_dir=dataset_path, seed=int(cfg["seed"]))

    tasks = load_tasks(dataset_path, cfg["split"])
    by_agent: dict[str, Any] = {}
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    for agent_name in cfg["agents"]:
        agent = build_agent(agent_name)
        episodes = [run_episode(task, agent) for task in tasks]
        scored = score_episodes(episodes)
        trace_rows = [episode_to_trace_row(result) for result in episodes]
        by_agent[agent_name] = scored
        write_json(out_dir / f"{agent_name}.json", scored)
        _write_jsonl(traces_dir / f"{agent_name}.jsonl", trace_rows)

    combined = {
        "config": cfg,
        "n_tasks": len(tasks),
        "by_agent": by_agent,
    }
    report_path = out_dir / "results.json"
    write_json(report_path, combined)

    table_paths = build_result_tables(combined, out_dir)
    plot_paths = build_plot_data(combined, out_dir)

    return {
        "report": str(report_path),
        "tables": table_paths,
        "plots": plot_paths,
    }


def build_result_tables(results: dict[str, Any], out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    table1_rows: list[dict[str, Any]] = []
    for agent_name, payload in results["by_agent"].items():
        agg = payload["aggregate"]
        table1_rows.append(
            {
                "agent": agent_name,
                "TaskSuccess": round(agg["TaskSuccess"], 4),
                "PolicyViolations": round(agg["PolicyViolations"], 4),
                "RecoverySuccess": round(agg["RecoverySuccess"], 4),
                "ToolCallsUsed": round(agg["ToolCallsUsed"], 4),
            }
        )

    table2_rows: list[dict[str, Any]] = []
    for agent_name, payload in results["by_agent"].items():
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in payload["per_task"]:
            fault = row.get("PrimaryFault", "clean")
            grouped.setdefault(fault, []).append(row)

        for fault, rows in grouped.items():
            n = len(rows)
            table2_rows.append(
                {
                    "agent": agent_name,
                    "fault": fault,
                    "n": n,
                    "TaskSuccess": round(sum(r["TaskSuccess"] for r in rows) / n, 4),
                    "RecoverySuccess": round(sum(r["RecoverySuccess"] for r in rows) / n, 4),
                    "PolicyViolations": round(sum(r["PolicyViolations"] for r in rows) / n, 4),
                }
            )

    table1_path = out / "table1_overall.csv"
    table2_path = out / "table2_fault_breakdown.csv"
    _write_csv(table1_path, table1_rows)
    _write_csv(table2_path, table2_rows)

    md_path = out / "tables.md"
    md = []
    md.append("## Table 1: Overall Metrics by Baseline")
    md.extend(_markdown_table(table1_rows))
    md.append("")
    md.append("## Table 2: Per-Fault Breakdown")
    md.extend(_markdown_table(table2_rows))
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    return {
        "table1_csv": str(table1_path),
        "table2_csv": str(table2_path),
        "tables_md": str(md_path),
    }


def build_plot_data(results: dict[str, Any], out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig1_rows: list[dict[str, Any]] = []
    for agent_name, payload in results["by_agent"].items():
        caps = payload["budgeted_success"]["caps"]
        success = payload["budgeted_success"]["success"]
        auc = payload["budgeted_success"]["auc"]
        for cap, succ in zip(caps, success):
            fig1_rows.append({"agent": agent_name, "cap": cap, "success": succ, "auc": auc})

    fig2_rows: list[dict[str, Any]] = []
    for agent_name, payload in results["by_agent"].items():
        for row in payload["per_task"]:
            if row["TimeToRecovery"] is None:
                continue
            fig2_rows.append({"agent": agent_name, "time_to_recovery": row["TimeToRecovery"]})

    fig1_path = out / "figure1_budgeted_success.csv"
    fig2_path = out / "figure2_time_to_recovery.csv"
    _write_csv(fig1_path, fig1_rows)
    _write_csv(fig2_path, fig2_rows)

    return {
        "figure1_csv": str(fig1_path),
        "figure2_csv": str(fig2_path),
    }


def _normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    merged = dict(DEFAULT_CONFIG)
    merged.update(config)
    agents = merged.get("agents")
    if isinstance(agents, str):
        merged["agents"] = [agents]
    return merged


def _parse_simple_yaml(raw: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_list_key: str | None = None
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and current_list_key is not None:
            data.setdefault(current_list_key, []).append(_parse_scalar(stripped[2:].strip()))
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            current_list_key = key
            data[key] = []
            continue
        current_list_key = None
        data[key] = _parse_scalar(value)
    return data


def _parse_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        parts = [part.strip().strip("\"'") for part in value[1:-1].split(",") if part.strip()]
        return parts
    if value.isdigit():
        return int(value)
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return value.strip("\"'")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["(no rows)"]
    columns = list(rows[0].keys())
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row[col]) for col in columns) + " |" for row in rows]
    return [header, divider, *body]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
