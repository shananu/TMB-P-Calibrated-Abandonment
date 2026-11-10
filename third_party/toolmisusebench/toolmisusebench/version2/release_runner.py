from __future__ import annotations

import json
import platform
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from toolmisusebench.baselines import build_agent
from toolmisusebench.dataset import load_tasks
from toolmisusebench.harness.metrics import episode_to_trace_row
from toolmisusebench.harness.runner import EpisodeResult, run_episode
from toolmisusebench.harness.scoring import score_episodes
from toolmisusebench.types import Task
from toolmisusebench.version2.generator import generate_dataset_v2
from toolmisusebench.version2.llm_agent import OpenAICompatibleToolAgent
from toolmisusebench.version2.quality import analyze_dataset_v2


DEFAULT_DETERMINISTIC_AGENTS = (
    "heuristic",
    "schema_repair",
    "policy_aware",
    "instruction_parser_no_recovery",
    "instruction_parser",
)
DEFAULT_BUDGET_CAPS = (2, 3, 4, 6, 8, 10)


class ReleaseRun:
    def __init__(self, out_dir: str | Path, config: dict[str, Any]) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.status_path = self.out_dir / "run_status.json"
        self.events_path = self.out_dir / "run_events.jsonl"
        self.manifest_path = self.out_dir / "run_manifest.json"
        self.status = self._load_status()
        if not self.manifest_path.exists():
            self._write_json(
                self.manifest_path,
                {
                    "created_at": _now(),
                    "platform": platform.platform(),
                    "python": sys.version,
                    "config": config,
                },
            )

    def start(self) -> None:
        self.status["state"] = "running"
        self.status["started_at"] = self.status.get("started_at", _now())
        self._save_status()
        self.event("run", "started", "v2 release run started")

    def complete(self) -> None:
        self.status["state"] = "complete"
        self.status["completed_at"] = _now()
        self._save_status()
        self.event("run", "complete", "v2 release run completed")

    def waiting(self, message: str) -> None:
        self.status["state"] = "waiting_for_decision"
        self.status["message"] = message
        self._save_status()
        self.event("run", "waiting_for_decision", message)

    def fail(self, stage: str, error: Exception) -> None:
        self.stage(stage, "failed", error=str(error))
        self.status["state"] = "failed"
        self.status["message"] = str(error)
        self._save_status()
        self.event(stage, "failed", str(error))

    def stage(self, name: str, state: str, **fields: Any) -> None:
        stages = self.status.setdefault("stages", {})
        stage = stages.setdefault(name, {})
        stage.update(fields)
        stage["state"] = state
        stage["updated_at"] = _now()
        if state == "running":
            stage.setdefault("started_at", _now())
        if state in {"complete", "failed", "skipped"}:
            stage["completed_at"] = _now()
        self._save_status()

    def event(self, stage: str, state: str, message: str, **fields: Any) -> None:
        payload = {"at": _now(), "stage": stage, "state": state, "message": message, **fields}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")

    def _load_status(self) -> dict[str, Any]:
        if self.status_path.exists():
            return json.loads(self.status_path.read_text(encoding="utf-8"))
        return {"state": "pending", "stages": {}, "updated_at": _now()}

    def _save_status(self) -> None:
        self.status["updated_at"] = _now()
        self._write_json(self.status_path, self.status)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp.replace(path)


def run_v2_release(
    out_dir: str | Path,
    dataset_dir: str | Path,
    seed: int,
    llm_base_url: str | None,
    llm_model: str,
    calibration_tasks: int = 48,
    llm_panel_tasks: int = 216,
    llm_sweep_tasks_per_cap: int = 48,
    llm_budget_caps: tuple[int, ...] = (2, 4, 6, 10),
    llm_max_tokens: int = 128,
    llm_disable_thinking: bool = True,
    llm_progress_every: int = 5,
    max_llm_minutes: float = 180.0,
    force_llm_panel: bool = False,
) -> dict[str, Any]:
    config = {
        "version": "v0.2",
        "seed": seed,
        "dataset_dir": str(dataset_dir),
        "size_profile": "large",
        "deterministic_agents": list(DEFAULT_DETERMINISTIC_AGENTS),
        "llm": {
            "base_url": llm_base_url,
            "model": llm_model,
            "calibration_tasks": calibration_tasks,
            "panel_tasks": llm_panel_tasks,
            "sweep_tasks_per_cap": llm_sweep_tasks_per_cap,
            "budget_caps": list(llm_budget_caps),
            "max_tokens": llm_max_tokens,
            "disable_thinking": llm_disable_thinking,
            "progress_every": llm_progress_every,
            "max_minutes": max_llm_minutes,
            "force_panel": force_llm_panel,
        },
    }
    run = ReleaseRun(out_dir, config)
    try:
        run.start()
        dataset_path = Path(dataset_dir)
        if not (dataset_path / "v0_2_freeze.json").exists():
            run.stage("dataset", "running", message="Generating v2 large dataset")
            payload = generate_dataset_v2("v0.2", dataset_path, seed=seed, size_profile="large")
            run.stage("dataset", "complete", **payload)
            run.event("dataset", "complete", "Generated v2 large dataset", **payload)
        else:
            run.stage("dataset", "skipped", message="Existing v2 dataset retained", dataset=str(dataset_path))

        quality = analyze_dataset_v2(dataset_path, ["test_public"])
        ReleaseRun._write_json(Path(out_dir) / "dataset_quality_report.json", quality)
        tasks = load_tasks(dataset_path, "test_public")

        for agent_name in DEFAULT_DETERMINISTIC_AGENTS:
            _run_agent_stage(
                run=run,
                stage_name=f"deterministic_{agent_name}",
                tasks=tasks,
                agent_factory=lambda name=agent_name: build_agent(name),
                progress_every=50,
            )

        _write_fault_ablation_summary(run, DEFAULT_DETERMINISTIC_AGENTS)
        _run_deterministic_budget_sweeps(run, tasks)

        if llm_base_url is None:
            run.stage("llm_calibration", "skipped", message="No local LLM server URL was supplied")
            run.stage("llm_balanced_panel", "skipped", message="No local LLM server URL was supplied")
            run.complete()
            return _write_release_summary(run)

        calibration = select_balanced_tasks(tasks, calibration_tasks)
        calibration_score = _run_agent_stage(
            run=run,
            stage_name="llm_calibration",
            tasks=calibration,
            agent_factory=lambda: OpenAICompatibleToolAgent(
                base_url=llm_base_url,
                model=llm_model,
                max_tokens=llm_max_tokens,
                disable_thinking=llm_disable_thinking,
            ),
            progress_every=llm_progress_every,
        )
        elapsed = float(calibration_score["stage_metadata"]["elapsed_seconds"])
        llm_total_tasks = llm_panel_tasks + (llm_sweep_tasks_per_cap * len(llm_budget_caps))
        projected_minutes = (elapsed / max(1, len(calibration))) * llm_total_tasks / 60.0
        run.stage(
            "llm_calibration",
            "complete",
            projected_panel_minutes=round(projected_minutes, 2),
            message="Calibration complete",
        )

        if projected_minutes > max_llm_minutes and not force_llm_panel:
            run.stage(
                "llm_balanced_panel",
                "skipped",
                message=(
                    f"Projected {projected_minutes:.1f} minutes exceeds the configured limit "
                    f"of {max_llm_minutes:.1f} minutes. Re-run with --force-llm-panel to continue."
                ),
            )
            run.waiting("LLM calibration exceeded the configured runtime limit.")
            return _write_release_summary(run)

        panel = select_balanced_tasks(tasks, llm_panel_tasks)
        _run_agent_stage(
            run=run,
            stage_name="llm_balanced_panel",
            tasks=panel,
            agent_factory=lambda: OpenAICompatibleToolAgent(base_url=llm_base_url, model=llm_model),
            progress_every=1,
        )
        _run_llm_budget_sweeps(
            run=run,
            tasks=tasks,
            base_url=llm_base_url,
            model=llm_model,
            tasks_per_cap=llm_sweep_tasks_per_cap,
            caps=llm_budget_caps,
            max_tokens=llm_max_tokens,
            disable_thinking=llm_disable_thinking,
            progress_every=llm_progress_every,
        )
        run.complete()
        return _write_release_summary(run)
    except Exception as exc:
        active_stage = _active_stage(run.status)
        run.fail(active_stage, exc)
        raise


def select_balanced_tasks(tasks: list[Task], count: int) -> list[Task]:
    if count <= 0:
        raise ValueError("count must be > 0")
    groups: dict[tuple[Any, ...], list[Task]] = defaultdict(list)
    for task in tasks:
        groups[_task_slice(task)].append(task)

    keys = sorted(groups)
    selected: list[Task] = []
    if count <= len(keys):
        for index in range(count):
            key = keys[(index * len(keys)) // count]
            selected.append(groups[key][0])
        return selected

    for key in keys:
        selected.append(groups[key][0])

    round_index = 1
    while len(selected) < min(count, len(tasks)):
        added = False
        for key in keys:
            if len(selected) >= count:
                break
            if round_index < len(groups[key]):
                selected.append(groups[key][round_index])
                added = True
        if not added:
            break
        round_index += 1
    return selected


def _run_deterministic_budget_sweeps(run: ReleaseRun, tasks: list[Task]) -> None:
    rows: list[dict[str, Any]] = []
    for agent_name in ("instruction_parser_no_recovery", "instruction_parser"):
        for cap in DEFAULT_BUDGET_CAPS:
            stage_name = f"budget_sweep_{agent_name}_cap_{cap}"
            score = _run_agent_stage(
                run=run,
                stage_name=stage_name,
                tasks=_with_budget_cap(tasks, cap),
                agent_factory=lambda name=agent_name: build_agent(name),
                progress_every=50,
            )
            rows.append(
                {
                    "agent": agent_name,
                    "cap": cap,
                    "TaskSuccess": score["aggregate"]["TaskSuccess"],
                    "RecoverySuccess": score["aggregate"]["RecoverySuccess"],
                    "ToolCallsUsed": score["aggregate"]["ToolCallsUsed"],
                }
            )
    _write_json(Path(run.out_dir) / "budget_sweep_deterministic.json", {"rows": rows, "caps": list(DEFAULT_BUDGET_CAPS)})


def _run_llm_budget_sweeps(
    run: ReleaseRun,
    tasks: list[Task],
    base_url: str,
    model: str,
    tasks_per_cap: int,
    caps: tuple[int, ...],
    max_tokens: int,
    disable_thinking: bool,
    progress_every: int,
) -> None:
    rows: list[dict[str, Any]] = []
    panel = select_balanced_tasks(tasks, tasks_per_cap)
    for cap in caps:
        stage_name = f"llm_budget_sweep_cap_{cap}"
        score = _run_agent_stage(
            run=run,
            stage_name=stage_name,
            tasks=_with_budget_cap(panel, cap),
            agent_factory=lambda: OpenAICompatibleToolAgent(
                base_url=base_url,
                model=model,
                max_tokens=max_tokens,
                disable_thinking=disable_thinking,
            ),
            progress_every=progress_every,
        )
        rows.append(
            {
                "agent": "llm",
                "cap": cap,
                "TaskSuccess": score["aggregate"]["TaskSuccess"],
                "RecoverySuccess": score["aggregate"]["RecoverySuccess"],
                "ToolCallsUsed": score["aggregate"]["ToolCallsUsed"],
            }
        )
    _write_json(Path(run.out_dir) / "budget_sweep_llm.json", {"rows": rows, "caps": list(caps), "tasks_per_cap": tasks_per_cap})


def _write_fault_ablation_summary(run: ReleaseRun, agent_names: tuple[str, ...]) -> None:
    output: dict[str, Any] = {"agents": {}}
    for agent_name in agent_names:
        score_path = run.out_dir / "stages" / f"deterministic_{agent_name}" / "score.json"
        if not score_path.exists():
            continue
        payload = json.loads(score_path.read_text(encoding="utf-8"))
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in payload.get("per_task", []):
            grouped[str(row.get("PrimaryFault", "unknown"))].append(row)
        output["agents"][agent_name] = {
            fault: {
                "n": len(rows),
                "TaskSuccess": _mean(rows, "TaskSuccess"),
                "RecoverySuccess": _mean(rows, "RecoverySuccess"),
                "PolicyViolations": _mean(rows, "PolicyViolations"),
                "ToolCallsUsed": _mean(rows, "ToolCallsUsed"),
            }
            for fault, rows in sorted(grouped.items())
        }
    _write_json(run.out_dir / "fault_ablation_summary.json", output)


def _with_budget_cap(tasks: list[Task], cap: int) -> list[Task]:
    if cap <= 0:
        raise ValueError("Budget cap must be > 0")
    adjusted: list[Task] = []
    for task in tasks:
        budget = task.budget.model_copy(update={"max_steps": cap, "max_tool_calls": cap})
        adjusted.append(task.model_copy(update={"budget": budget}))
    return adjusted


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[key]) for row in rows) / len(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    ReleaseRun._write_json(path, payload)


def read_release_status(out_dir: str | Path) -> dict[str, Any]:
    path = Path(out_dir) / "run_status.json"
    if not path.exists():
        raise FileNotFoundError(f"No release-run status exists at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _run_agent_stage(
    run: ReleaseRun,
    stage_name: str,
    tasks: list[Task],
    agent_factory: Callable[[], Any],
    progress_every: int,
) -> dict[str, Any]:
    stage_dir = run.out_dir / "stages" / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    score_path = stage_dir / "score.json"
    trace_path = stage_dir / "episodes.jsonl"
    if score_path.exists():
        run.stage(stage_name, "skipped", message="Completed artifact already exists", artifact=str(score_path))
        return json.loads(score_path.read_text(encoding="utf-8"))

    completed = _load_completed_results(trace_path)
    pending = [task for task in tasks if task.task_id not in completed]
    run.stage(stage_name, "running", completed=len(completed), total=len(tasks), message="Stage running")
    run.event(stage_name, "running", "Stage started", completed=len(completed), total=len(tasks))
    started = time.monotonic()
    agent = agent_factory()

    with trace_path.open("a", encoding="utf-8") as handle:
        for index, task in enumerate(pending, start=1):
            result = run_episode(task, agent)
            completed[result.task_id] = result
            handle.write(json.dumps(episode_to_trace_row(result), sort_keys=True))
            handle.write("\n")
            handle.flush()

            total_completed = len(completed)
            if index % progress_every == 0 or total_completed == len(tasks):
                elapsed = time.monotonic() - started
                rate = elapsed / max(index, 1)
                remaining = rate * (len(tasks) - total_completed)
                run.stage(
                    stage_name,
                    "running",
                    completed=total_completed,
                    total=len(tasks),
                    elapsed_seconds=round(elapsed, 3),
                    estimated_remaining_seconds=round(remaining, 3),
                )
                run.event(
                    stage_name,
                    "progress",
                    "Stage progress updated",
                    completed=total_completed,
                    total=len(tasks),
                    estimated_remaining_seconds=round(remaining, 3),
                )

    results = [completed[task.task_id] for task in tasks]
    score = score_episodes(results)
    metadata = {
        "stage": stage_name,
        "task_count": len(tasks),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "completed_at": _now(),
    }
    score["stage_metadata"] = metadata
    ReleaseRun._write_json(score_path, score)
    ReleaseRun._write_json(stage_dir / "metadata.json", metadata)
    run.stage(stage_name, "complete", artifact=str(score_path), **metadata)
    run.event(
        stage_name,
        "complete",
        "Stage completed",
        task_count=metadata["task_count"],
        elapsed_seconds=metadata["elapsed_seconds"],
        completed_at=metadata["completed_at"],
    )
    return score


def _load_completed_results(trace_path: Path) -> dict[str, EpisodeResult]:
    if not trace_path.exists():
        return {}
    results: dict[str, EpisodeResult] = {}
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            result = EpisodeResult(**row)
            results[result.task_id] = result
    return results


def _task_slice(task: Task) -> tuple[Any, ...]:
    primary_fault = task.fault_plan[0].fault_type if task.fault_plan else "clean"
    budget = task.budget
    return (
        task.domain,
        primary_fault,
        task.difficulty,
        budget.max_steps,
        budget.max_tool_calls,
        budget.max_retries,
    )


def _write_release_summary(run: ReleaseRun) -> dict[str, Any]:
    stage_summaries: dict[str, Any] = {}
    for score_path in sorted((run.out_dir / "stages").glob("*/score.json")):
        payload = json.loads(score_path.read_text(encoding="utf-8"))
        stage_summaries[score_path.parent.name] = {
            "aggregate": payload.get("aggregate", {}),
            "budgeted_success": payload.get("budgeted_success", {}),
            "stage_metadata": payload.get("stage_metadata", {}),
        }
    summary = {"status": run.status, "stages": stage_summaries, "updated_at": _now()}
    ReleaseRun._write_json(run.out_dir / "release_summary.json", summary)
    return summary


def _active_stage(status: dict[str, Any]) -> str:
    for name, stage in status.get("stages", {}).items():
        if stage.get("state") == "running":
            return str(name)
    return "run"


def _now() -> str:
    return datetime.now(UTC).isoformat()
