from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from random import Random
from typing import Any
from importlib.util import find_spec

from toolmisusebench.envs.crud_env import CrudEnv
from toolmisusebench.envs.files_env import FilesEnv
from toolmisusebench.envs.retrieval_env import RetrievalEnv
from toolmisusebench.envs.scheduler_env import SchedulerEnv
from toolmisusebench.types import Budget, Task


DEFAULT_SPLIT_SIZES: dict[str, int] = {
    "train": 24,
    "dev": 12,
    "test_public": 12,
}

SIZE_PROFILES: dict[str, dict[str, int]] = {
    "default": dict(DEFAULT_SPLIT_SIZES),
    "small": {"train": 200, "dev": 50, "test_public": 50},
    "medium": {"train": 1000, "dev": 200, "test_public": 300},
    "large": {"train": 5000, "dev": 800, "test_public": 1000},
}

DOMAINS: tuple[str, ...] = ("crud", "retrieval", "files", "scheduling")

TOOLSET_BY_DOMAIN: dict[str, tuple[str, list[dict[str, Any]]]] = {
    "crud": (CrudEnv.TOOLSET_ID, CrudEnv.TOOL_SCHEMAS),
    "retrieval": (RetrievalEnv.TOOLSET_ID, RetrievalEnv.TOOL_SCHEMAS),
    "files": (FilesEnv.TOOLSET_ID, FilesEnv.TOOL_SCHEMAS),
    "scheduling": (SchedulerEnv.TOOLSET_ID, SchedulerEnv.TOOL_SCHEMAS),
}


def generate_dataset(
    version: str,
    out_dir: str | Path,
    seed: int,
    write_parquet: bool = True,
    split_sizes: dict[str, int] | None = None,
    size_profile: str = "default",
) -> dict[str, Any]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    resolved_split_sizes = resolve_split_sizes(
        split_sizes=split_sizes,
        size_profile=size_profile,
    )

    rng = Random(seed)
    generated_files: list[Path] = []
    split_counts: dict[str, int] = {}
    quality_issues: list[str] = []

    for split, size in resolved_split_sizes.items():
        tasks = [_build_task(split=split, index=i, rng=rng, base_seed=seed) for i in range(size)]
        quality_issues.extend(_validate_tasks(tasks))
        split_counts[split] = len(tasks)
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = split_dir / "tasks.jsonl"
        _write_jsonl(jsonl_path, [task.model_dump() for task in tasks])
        generated_files.append(jsonl_path)

        if write_parquet:
            parquet_path = split_dir / "tasks.parquet"
            if _try_write_parquet(parquet_path, [task.model_dump() for task in tasks]):
                generated_files.append(parquet_path)

    manifest = {
        "version": version,
        "seed": seed,
        "splits": split_counts,
        "files": [_file_entry(path, root) for path in generated_files],
    }

    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    freeze_path = root / "v0_1_freeze.json"
    freeze = {
        "version": version,
        "seed": seed,
        "domain_set": list(DOMAINS),
        "split_sizes": resolved_split_sizes,
        "size_profile": size_profile,
        "fault_types": ["schema_drift", "rate_limit", "timeout", "authz", "adversarial_error"],
    }
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True), encoding="utf-8")

    if quality_issues:
        preview = quality_issues[:10]
        raise ValueError(
            "Generated dataset failed coherence checks. "
            f"Total issues: {len(quality_issues)}. Sample: {preview}"
        )

    return {
        "out_dir": str(root),
        "manifest": str(manifest_path),
        "freeze": str(freeze_path),
        "splits": split_counts,
        "files_written": len(generated_files),
        "size_profile": size_profile,
        "quality_issues": 0,
    }


def resolve_split_sizes(
    split_sizes: dict[str, int] | None,
    size_profile: str,
) -> dict[str, int]:
    profile_name = size_profile.strip().lower()
    if profile_name not in SIZE_PROFILES:
        raise ValueError(
            f"Unknown size profile '{size_profile}'. Available: {', '.join(sorted(SIZE_PROFILES))}."
        )

    resolved = dict(SIZE_PROFILES[profile_name])
    if split_sizes is not None:
        for key in ("train", "dev", "test_public"):
            if key not in split_sizes:
                raise ValueError(f"split_sizes is missing required key '{key}'.")
            value = int(split_sizes[key])
            if value <= 0:
                raise ValueError(f"split size for '{key}' must be > 0.")
            resolved[key] = value
    return resolved


def _build_task(split: str, index: int, rng: Random, base_seed: int) -> Task:
    domain = DOMAINS[index % len(DOMAINS)]
    task_seed = (base_seed * 1000) + index + _split_offset(split)
    toolset_id, tool_schemas = TOOLSET_BY_DOMAIN[domain]
    scenario = _build_scenario(domain=domain, split=split, index=index, rng=rng)
    faults = _fault_plan_for(scenario["fault_tool"], index, rng)

    return Task(
        task_id=f"{split}_{domain}_{index:04d}",
        split=split,  # type: ignore[arg-type]
        difficulty=_difficulty_for(index),
        domain=domain,  # type: ignore[arg-type]
        instruction=scenario["instruction"],
        toolset_id=toolset_id,
        tool_schemas=tool_schemas,
        initial_state=scenario["initial_state"],
        success_criteria=scenario["success_criteria"],
        budget=Budget(max_steps=6, max_tool_calls=6, max_retries=2, timeout_ms=1000),
        fault_plan=faults,
        gold_summary=None,
        seed=task_seed,
    )


def _build_scenario(domain: str, split: str, index: int, rng: Random) -> dict[str, Any]:
    if domain == "crud":
        return _build_crud_scenario(split, index, rng)
    if domain == "retrieval":
        return _build_retrieval_scenario(split, index, rng)
    if domain == "files":
        return _build_files_scenario(split, index, rng)
    return _build_scheduler_scenario(split, index, rng)


def _build_crud_scenario(split: str, index: int, rng: Random) -> dict[str, Any]:
    account_a = f"a{split[:1]}{index:04d}"
    account_b = f"b{split[:1]}{index:04d}"
    tiers = ["basic", "plus", "premium", "enterprise"]
    current_tier = rng.choice(tiers)
    target_tier = rng.choice([tier for tier in tiers if tier != current_tier])
    op_mode = index % 3

    initial_state = {
        "accounts": {
            account_a: {"id": account_a, "status": "active", "tier": current_tier},
            account_b: {"id": account_b, "status": "active", "tier": rng.choice(tiers)},
        }
    }

    if op_mode == 0:
        reason = rng.choice(["user_request", "risk_review", "duplicate_account", "compliance"]) 
        instruction = rng.choice(
            [
                f"Close account {account_a} and mark the reason as {reason}.",
                f"Deactivate user account {account_a}; use reason {reason}.",
                f"Archive account {account_a} by closing it with reason {reason}.",
            ]
        )
        success_criteria = [
            {"op": "equals", "path": f"accounts.{account_a}.status", "value": "closed"},
            {"op": "equals", "path": f"accounts.{account_a}.closed_reason", "value": reason},
        ]
        fault_tool = "close_account"
    elif op_mode == 1:
        instruction = rng.choice(
            [
                f"Update account {account_a} so tier is {target_tier}.",
                f"Set the subscription tier for {account_a} to {target_tier}.",
                f"Patch account {account_a} with tier={target_tier}.",
            ]
        )
        success_criteria = [{"op": "equals", "path": f"accounts.{account_a}.tier", "value": target_tier}]
        fault_tool = "update_account"
    else:
        new_account_id = f"n{split[:1]}{index:04d}"
        instruction = rng.choice(
            [
                f"Create a new account with id {new_account_id}.",
                f"Register account {new_account_id} with active status.",
                f"Add customer account {new_account_id}.",
            ]
        )
        success_criteria = [
            {"op": "in", "path": "accounts", "value": new_account_id},
            {"op": "equals", "path": f"accounts.{new_account_id}.status", "value": "active"},
        ]
        fault_tool = "create_account"

    return {
        "instruction": instruction,
        "initial_state": initial_state,
        "success_criteria": success_criteria,
        "fault_tool": fault_tool,
    }


def _build_retrieval_scenario(split: str, index: int, rng: Random) -> dict[str, Any]:
    topic = rng.choice(["billing", "security", "latency", "compliance", "onboarding", "retention"])
    verb = rng.choice(["retrieve", "find", "look up", "fetch", "locate"])
    doc_target = f"d{split[:1]}{index:04d}"
    doc_alt = f"x{split[:1]}{index:04d}"
    query = f"{topic} incident {index % 17}"
    mode = index % 2

    docs = [
        {
            "id": doc_target,
            "title": f"{topic.title()} Incident Brief {index}",
            "text": f"{query} root cause and mitigation timeline",
        },
        {
            "id": doc_alt,
            "title": f"General Operations Note {index}",
            "text": "weekly housekeeping checklist and routine updates",
        },
    ]

    if mode == 0:
        instruction = rng.choice(
            [
                f"{verb.capitalize()} document {doc_target}.",
                f"Get the full contents for doc id {doc_target}.",
                f"Open retrieval record {doc_target}.",
            ]
        )
        success_criteria = [
            {"op": "successful_tool_call", "tool_name": "get_doc", "min_count": 1},
        ]
        fault_tool = "get_doc"
    else:
        instruction = rng.choice(
            [
                f"Search docs for '{query}' and inspect top results.",
                f"Run document search with query '{query}'.",
                f"Find relevant docs for '{query}'.",
            ]
        )
        success_criteria = [
            {"op": "successful_tool_call", "tool_name": "search_docs", "min_count": 1},
        ]
        fault_tool = "search_docs"

    return {
        "instruction": instruction,
        "initial_state": {"docs": docs},
        "success_criteria": success_criteria,
        "fault_tool": fault_tool,
    }


def _build_files_scenario(split: str, index: int, rng: Random) -> dict[str, Any]:
    folder = rng.choice(["reports", "drafts", "exports", "snapshots", "notes"])
    name = rng.choice(["summary", "status", "runbook", "rollup", "delta"])
    ext = rng.choice(["txt", "md", "log"])
    target_path = f"/{folder}/{name}_{split}_{index}.{ext}"
    content = f"entry-{split}-{index}-{rng.randint(100, 999)}"
    mode = index % 2

    initial_state = {
        "files": {
            f"/{folder}/README.md": "team workspace",
            f"/{folder}/seed_{index % 7}.txt": f"seed {index}",
        }
    }

    if mode == 0:
        instruction = rng.choice(
            [
                f"Write '{content}' to {target_path}.",
                f"Create file {target_path} with content '{content}'.",
                f"Store the text '{content}' at {target_path}.",
            ]
        )
        success_criteria = [
            {"op": "in", "path": "files", "value": target_path},
            {"op": "map_value_equals", "path": "files", "key": target_path, "value": content},
        ]
        fault_tool = "write_file"
    else:
        existing_path = f"/{folder}/README.md"
        instruction = rng.choice(
            [
                f"Read file {existing_path}.",
                f"Open and inspect {existing_path}.",
                f"Fetch content from {existing_path}.",
            ]
        )
        success_criteria = [
            {"op": "successful_tool_call", "tool_name": "read_file", "min_count": 1},
        ]
        fault_tool = "read_file"

    return {
        "instruction": instruction,
        "initial_state": initial_state,
        "success_criteria": success_criteria,
        "fault_tool": fault_tool,
    }


def _build_scheduler_scenario(split: str, index: int, rng: Random) -> dict[str, Any]:
    name = rng.choice(["backup", "cleanup", "reindex", "sync", "archive"]) + f"_{split}_{index}"
    cron = rng.choice(["0 * * * *", "15 * * * *", "0 0 * * *", "30 2 * * *", "5 4 * * *"])
    command = rng.choice(["run_backup", "sync_data", "rotate_logs", "refresh_index", "export_metrics"])
    mode = index % 2

    initial_state = {
        "jobs": [
            {
                "name": f"seed_job_{split}_{index % 5}",
                "cron": "0 * * * *",
                "command": "seed_task",
            }
        ]
    }

    if mode == 0:
        instruction = rng.choice(
            [
                f"Create scheduler job {name} with cron '{cron}' and command '{command}'.",
                f"Register job {name}; cron={cron}; command={command}.",
                f"Add a cron job named {name} to run {command} on '{cron}'.",
            ]
        )
        success_criteria = [
            {"op": "successful_tool_call", "tool_name": "create_job", "min_count": 1},
            {"op": "in", "path": "jobs", "value": {"name": name, "cron": cron, "command": command}},
        ]
        fault_tool = "create_job"
    else:
        instruction = rng.choice(
            [
                f"Validate cron expression '{cron}'.",
                f"Check whether '{cron}' is a valid cron schedule.",
                f"Run cron validation for '{cron}'.",
            ]
        )
        success_criteria = [
            {"op": "successful_tool_call", "tool_name": "validate_cron", "min_count": 1},
        ]
        fault_tool = "validate_cron"

    return {
        "instruction": instruction,
        "initial_state": initial_state,
        "success_criteria": success_criteria,
        "fault_tool": fault_tool,
    }


def _split_offset(split: str) -> int:
    if split == "train":
        return 0
    if split == "dev":
        return 10_000
    return 20_000


def _difficulty_for(index: int) -> str:
    if index % 3 == 0:
        return "easy"
    if index % 3 == 1:
        return "medium"
    return "hard"


def _fault_plan_for(tool_name: str, index: int, rng: Random) -> list[dict[str, Any]]:
    mode = index % 6
    if mode == 0:
        return []
    if mode == 1:
        return [
            {
                "fault_type": "timeout",
                "trigger": {"on_tool": tool_name, "on_nth_call": 1},
                "severity": "soft",
                "payload": {"timeout_ms": 50},
            }
        ]
    if mode == 2:
        return [
            {
                "fault_type": "rate_limit",
                "trigger": {"on_tool": tool_name, "prob": {"p": 0.6, "seed": 777 + index}},
                "severity": "soft",
                "payload": {"after_n_calls": 0, "retry_after_ms": 100},
            }
        ]
    if mode == 3:
        rename_args: dict[str, str] = {"doc_id": "id"}
        if tool_name == "close_account":
            rename_args = {"id": "account_id"}
        elif tool_name == "write_file":
            rename_args = {"path": "file_path"}
        elif tool_name == "create_job":
            rename_args = {"command": "cmd"}
        elif tool_name == "validate_cron":
            rename_args = {"cron": "schedule"}
        elif tool_name == "search_docs":
            rename_args = {"query": "q"}
        elif tool_name == "read_file":
            rename_args = {"path": "file_path"}
        return [
            {
                "fault_type": "schema_drift",
                "trigger": {"on_tool": tool_name, "on_nth_call": 1},
                "severity": "hard",
                "payload": {"rename_args": rename_args},
            }
        ]
    if mode == 4:
        return [
            {
                "fault_type": "authz",
                "trigger": {"on_tool": tool_name},
                "severity": "hard",
                "payload": {"deny": rng.choice([True, False]), "required_role": "admin"},
            }
        ]
    return [
        {
            "fault_type": "timeout",
            "trigger": {"on_tool": tool_name, "on_nth_call": 1},
            "severity": "soft",
            "payload": {"timeout_ms": 50},
        },
        {
            "fault_type": "adversarial_error",
            "trigger": {"on_tool": tool_name, "on_nth_call": 1},
            "severity": "soft",
            "payload": {"message": "Internal issue. Please retry.", "code": "ambiguous_error"},
        },
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _try_write_parquet(path: Path, rows: list[dict[str, Any]]) -> bool:
    pa = find_spec("pyarrow")
    if pa is None:
        return False
    try:
        pq = find_spec("pyarrow.parquet")
    except ModuleNotFoundError:
        pq = None
    if pq is None:
        return False

    pyarrow = importlib.import_module("pyarrow")
    parquet = importlib.import_module("pyarrow.parquet")

    table = pyarrow.Table.from_pylist(rows)
    parquet.write_table(table, path)
    return True


def _file_entry(path: Path, root: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": digest,
    }


def _validate_tasks(tasks: list[Task]) -> list[str]:
    issues: list[str] = []
    for task in tasks:
        issues.extend(_validate_task_coherence(task))
    return issues


def _validate_task_coherence(task: Task) -> list[str]:
    issues: list[str] = []
    available_tools = {schema.get("name") for schema in task.tool_schemas if isinstance(schema.get("name"), str)}

    if not task.instruction.strip():
        issues.append(f"{task.task_id}: empty instruction")
    if not task.success_criteria:
        issues.append(f"{task.task_id}: missing success criteria")

    if task.domain == "crud" and "accounts" not in task.initial_state:
        issues.append(f"{task.task_id}: crud task missing accounts in initial_state")
    if task.domain == "retrieval" and "docs" not in task.initial_state:
        issues.append(f"{task.task_id}: retrieval task missing docs in initial_state")
    if task.domain == "files" and "files" not in task.initial_state:
        issues.append(f"{task.task_id}: files task missing files in initial_state")
    if task.domain == "scheduling" and "jobs" not in task.initial_state:
        issues.append(f"{task.task_id}: scheduling task missing jobs in initial_state")

    for idx, criterion in enumerate(task.success_criteria):
        op = criterion.get("op")
        if op in {"equals", "exists", "in", "map_value_equals"} and "path" not in criterion:
            issues.append(f"{task.task_id}: criterion[{idx}] missing path for op={op}")
        if op in {"successful_tool_call", "tool_call"}:
            tool_name = criterion.get("tool_name")
            if not isinstance(tool_name, str):
                issues.append(f"{task.task_id}: criterion[{idx}] missing tool_name for op={op}")
            elif tool_name not in available_tools:
                issues.append(f"{task.task_id}: criterion[{idx}] references unknown tool '{tool_name}'")
        if op == "map_value_equals" and "key" not in criterion:
            issues.append(f"{task.task_id}: criterion[{idx}] missing key for map_value_equals")

    tool_props: dict[str, set[str]] = {}
    for schema in task.tool_schemas:
        name = schema.get("name")
        properties = schema.get("properties", {})
        if isinstance(name, str) and isinstance(properties, dict):
            tool_props[name] = set(properties.keys())

    for fidx, fault in enumerate(task.fault_plan):
        trigger = fault.trigger
        trigger_tool = trigger.get("on_tool")
        if trigger_tool is not None and trigger_tool not in available_tools:
            issues.append(f"{task.task_id}: fault[{fidx}] references unknown on_tool '{trigger_tool}'")
        if fault.fault_type == "schema_drift" and isinstance(trigger_tool, str):
            rename_args = fault.payload.get("rename_args", {})
            if isinstance(rename_args, dict):
                known = tool_props.get(trigger_tool, set())
                for old_name in rename_args:
                    if old_name not in known:
                        issues.append(
                            f"{task.task_id}: fault[{fidx}] schema_drift rename_args unknown field '{old_name}' "
                            f"for tool '{trigger_tool}'"
                        )

    return issues
