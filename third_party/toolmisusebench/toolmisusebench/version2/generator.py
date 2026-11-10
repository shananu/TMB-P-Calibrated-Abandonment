from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path
from random import Random
from typing import Any

from toolmisusebench.generator.core import TOOLSET_BY_DOMAIN, _build_scenario
from toolmisusebench.types import Budget, Task


V2_SPLIT_SIZES: dict[str, int] = {
    "train": 6000,
    "dev": 900,
    "test_public": 1200,
}

V2_SIZE_PROFILES: dict[str, dict[str, int]] = {
    "default": {"train": 432, "dev": 216, "test_public": 216},
    "small": {"train": 648, "dev": 216, "test_public": 216},
    "medium": {"train": 2400, "dev": 360, "test_public": 600},
    "large": dict(V2_SPLIT_SIZES),
}

DOMAINS: tuple[str, ...] = ("crud", "retrieval", "files", "scheduling")
DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")
FAULT_FAMILIES: tuple[str, ...] = (
    "clean",
    "timeout",
    "rate_limit",
    "schema_drift",
    "authz",
    "adversarial_error",
)
BUDGET_PROFILES: dict[str, Budget] = {
    "tight": Budget(max_steps=3, max_tool_calls=3, max_retries=1, timeout_ms=750),
    "standard": Budget(max_steps=6, max_tool_calls=6, max_retries=2, timeout_ms=1000),
    "relaxed": Budget(max_steps=10, max_tool_calls=10, max_retries=3, timeout_ms=1500),
}


def generate_dataset_v2(
    version: str,
    out_dir: str | Path,
    seed: int,
    split_sizes: dict[str, int] | None = None,
    size_profile: str = "default",
) -> dict[str, Any]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    resolved_split_sizes = resolve_split_sizes_v2(split_sizes=split_sizes, size_profile=size_profile)

    generated_files: list[Path] = []
    split_counts: dict[str, int] = {}
    for split, size in resolved_split_sizes.items():
        rows = [
            _build_task_v2(split=split, index=index, base_seed=seed)
            for index in range(size)
        ]
        split_counts[split] = len(rows)
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = split_dir / "tasks.jsonl"
        _write_jsonl(jsonl_path, [task.model_dump() for task in rows])
        generated_files.append(jsonl_path)

    manifest_path = root / "manifest.json"
    manifest = {
        "version": version,
        "seed": seed,
        "splits": split_counts,
        "files": [_file_entry(path, root) for path in generated_files],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    freeze_path = root / "v0_2_freeze.json"
    freeze = {
        "version": version,
        "seed": seed,
        "domain_set": list(DOMAINS),
        "difficulty_set": list(DIFFICULTIES),
        "fault_families": list(FAULT_FAMILIES),
        "budget_profiles": {name: budget.model_dump() for name, budget in BUDGET_PROFILES.items()},
        "split_sizes": resolved_split_sizes,
        "size_profile": size_profile,
        "design": "crossed_domain_fault_difficulty_budget",
    }
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "out_dir": str(root),
        "manifest": str(manifest_path),
        "freeze": str(freeze_path),
        "splits": split_counts,
        "files_written": len(generated_files),
        "size_profile": size_profile,
        "quality_issues": 0,
    }


def resolve_split_sizes_v2(
    split_sizes: dict[str, int] | None,
    size_profile: str,
) -> dict[str, int]:
    profile_name = size_profile.strip().lower()
    if profile_name not in V2_SIZE_PROFILES:
        raise ValueError(
            f"Unknown v2 size profile '{size_profile}'. Available: {', '.join(sorted(V2_SIZE_PROFILES))}."
        )

    resolved = dict(V2_SIZE_PROFILES[profile_name])
    if split_sizes is not None:
        for key in ("train", "dev", "test_public"):
            if key not in split_sizes:
                raise ValueError(f"split_sizes is missing required key '{key}'.")
            value = int(split_sizes[key])
            if value <= 0:
                raise ValueError(f"split size for '{key}' must be > 0.")
            resolved[key] = value
    return resolved


def _build_task_v2(split: str, index: int, base_seed: int) -> Task:
    combos = list(product(DOMAINS, FAULT_FAMILIES, DIFFICULTIES, tuple(BUDGET_PROFILES)))
    domain, fault_family, difficulty, budget_profile = combos[index % len(combos)]
    cycle = index // len(combos)
    scenario_index = (cycle * 97) + (index * 17) + _split_offset(split)
    rng = Random((base_seed * 100_003) + scenario_index)

    toolset_id, tool_schemas = TOOLSET_BY_DOMAIN[domain]
    scenario = _build_scenario(domain=domain, split=split, index=scenario_index, rng=rng)
    criteria = _strengthen_success_criteria(domain, scenario)
    fault_plan = _fault_plan_v2(scenario["fault_tool"], fault_family, index)

    return Task(
        task_id=f"{split}_v2_{domain}_{fault_family}_{difficulty}_{budget_profile}_{index:05d}",
        split=split,  # type: ignore[arg-type]
        difficulty=difficulty,  # type: ignore[arg-type]
        domain=domain,  # type: ignore[arg-type]
        instruction=scenario["instruction"],
        toolset_id=toolset_id,
        tool_schemas=tool_schemas,
        initial_state=scenario["initial_state"],
        success_criteria=criteria,
        budget=BUDGET_PROFILES[budget_profile],
        fault_plan=fault_plan,
        gold_summary=f"v2_profile={budget_profile}; fault_family={fault_family}",
        seed=(base_seed * 1000) + index + _split_offset(split),
    )


def _strengthen_success_criteria(domain: str, scenario: dict[str, Any]) -> list[dict[str, Any]]:
    criteria = list(scenario["success_criteria"])
    fault_tool = scenario["fault_tool"]
    initial_state = scenario["initial_state"]

    if domain == "retrieval":
        target_doc = initial_state["docs"][0]
        if fault_tool == "get_doc":
            return [
                {"op": "successful_tool_call", "tool_name": "get_doc", "min_count": 1},
                {
                    "op": "tool_output_path_equals",
                    "tool_name": "get_doc",
                    "output_path": "doc.id",
                    "value": target_doc["id"],
                },
            ]
        return [
            {"op": "successful_tool_call", "tool_name": "search_docs", "min_count": 1},
            {
                "op": "tool_output_path_equals",
                "tool_name": "search_docs",
                "output_path": "results.0.id",
                "value": target_doc["id"],
            },
        ]

    if domain == "files" and fault_tool == "read_file":
        return [
            {"op": "successful_tool_call", "tool_name": "read_file", "min_count": 1},
            {
                "op": "tool_output_path_equals",
                "tool_name": "read_file",
                "output_path": "content",
                "value": "team workspace",
            },
        ]

    if domain == "scheduling" and fault_tool == "validate_cron":
        return [
            {"op": "successful_tool_call", "tool_name": "validate_cron", "min_count": 1},
            {
                "op": "tool_output_path_equals",
                "tool_name": "validate_cron",
                "output_path": "valid",
                "value": True,
            },
        ]

    return criteria


def _fault_plan_v2(tool_name: str, fault_family: str, index: int) -> list[dict[str, Any]]:
    if fault_family == "clean":
        return []
    if fault_family == "timeout":
        return [
            {
                "fault_type": "timeout",
                "trigger": {"on_tool": tool_name, "on_nth_call": 1},
                "severity": "soft",
                "payload": {"timeout_ms": 50, "recoverability": "transient"},
            }
        ]
    if fault_family == "rate_limit":
        return [
            {
                "fault_type": "rate_limit",
                "trigger": {"on_tool": tool_name, "on_nth_call": 1},
                "severity": "soft",
                "payload": {"after_n_calls": 0, "retry_after_ms": 100, "recoverability": "transient"},
            }
        ]
    if fault_family == "schema_drift":
        return [
            {
                "fault_type": "schema_drift",
                "trigger": {"on_tool": tool_name, "on_nth_call": 1},
                "severity": "hard",
                "payload": {
                    "rename_args": _rename_args_for_tool(tool_name),
                    "drift_visibility": "hidden_backend",
                    "recoverability": "repairable",
                },
            }
        ]
    if fault_family == "authz":
        return [
            {
                "fault_type": "authz",
                "trigger": {"on_tool": tool_name, "on_nth_call": 1},
                "severity": "hard",
                "payload": {"deny": True, "required_role": "admin", "recoverability": "transient_denial"},
            }
        ]
    if fault_family == "adversarial_error":
        return [
            {
                "fault_type": "adversarial_error",
                "trigger": {"on_tool": tool_name, "on_nth_call": 1},
                "severity": "soft",
                "payload": {
                    "message": "The tool failed for an unclear reason. Try a safer recovery.",
                    "code": "ambiguous_error",
                    "recoverability": "transient_with_ambiguous_feedback",
                },
            },
            {
                "fault_type": "timeout",
                "trigger": {"on_tool": tool_name, "on_nth_call": 1},
                "severity": "soft",
                "payload": {"timeout_ms": 50},
            },
        ]
    raise ValueError(f"Unsupported v2 fault family: {fault_family}")


def _rename_args_for_tool(tool_name: str) -> dict[str, str]:
    mapping = {
        "get_account": {"id": "account_id"},
        "update_account": {"id": "account_id"},
        "create_account": {"fields": "payload"},
        "close_account": {"id": "account_id"},
        "search_docs": {"query": "q"},
        "get_doc": {"doc_id": "id"},
        "read_file": {"path": "file_path"},
        "write_file": {"path": "file_path"},
        "list_dir": {"path": "dir_path"},
        "create_job": {"command": "cmd"},
        "validate_cron": {"cron": "schedule"},
    }
    return mapping.get(tool_name, {})


def _split_offset(split: str) -> int:
    if split == "train":
        return 0
    if split == "dev":
        return 10_000
    return 20_000


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _file_entry(path: Path, root: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": digest,
    }
