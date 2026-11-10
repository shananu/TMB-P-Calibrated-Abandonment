from __future__ import annotations

import argparse

import json

from toolmisusebench.agent_loader import AgentLoadError, build_runtime_agent
from toolmisusebench.dataset import load_tasks
from toolmisusebench.experiment_pipeline import load_experiment_config, run_experiment_pipeline
from toolmisusebench.generator import SIZE_PROFILES, generate_dataset
from toolmisusebench.harness.artifacts import write_evaluation_artifacts
from toolmisusebench.harness.metrics import episode_to_trace_row
from toolmisusebench.harness.runner import run_episode
from toolmisusebench.harness.scoring import score_episodes
from toolmisusebench.version2.generator import V2_SIZE_PROFILES, generate_dataset_v2
from toolmisusebench.version2.release_runner import read_release_status, run_v2_release


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="toolmisusebench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate benchmark dataset artifacts")
    generate.add_argument("--version", required=True)
    generate.add_argument("--out", required=True)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--no-parquet", action="store_true")
    generate.add_argument(
        "--size-profile",
        default="default",
        choices=sorted(set(SIZE_PROFILES) | set(V2_SIZE_PROFILES)),
        help="Dataset size profile. Use split overrides for custom counts.",
    )
    generate.add_argument("--train-size", type=int, help="Override train split size.")
    generate.add_argument("--dev-size", type=int, help="Override dev split size.")
    generate.add_argument("--test-size", type=int, help="Override test_public split size.")

    evaluate = subparsers.add_parser("eval", help="Evaluate an agent on a dataset split")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--split", required=True)
    agent_group = evaluate.add_mutually_exclusive_group(required=True)
    agent_group.add_argument("--agent", help="Built-in agent: heuristic, schema_repair, policy_aware, local_llm")
    agent_group.add_argument("--agent-module", help="Custom agent in format module:ClassName")
    evaluate.add_argument("--agent-kwargs", default="{}", help="JSON object for custom agent constructor kwargs")
    evaluate.add_argument("--report", required=True)

    reproduce = subparsers.add_parser("reproduce-paper", help="Run paper experiment pipeline")
    reproduce.add_argument("--config", required=True)

    release = subparsers.add_parser("run-v2-release", help="Run resumable v2 large and LLM experiments")
    release.add_argument("--out", required=True, help="Directory for live status and experiment artifacts")
    release.add_argument("--dataset", default="data/toolmisusebench_v0_2_large")
    release.add_argument("--seed", type=int, default=42)
    release.add_argument("--llm-base-url", default=None, help="Local OpenAI-compatible server URL, e.g. http://127.0.0.1:8080/v1")
    release.add_argument("--llm-model", default="Qwen3-4B-Q4_K_M")
    release.add_argument("--calibration-tasks", type=int, default=48)
    release.add_argument("--llm-panel-tasks", type=int, default=216)
    release.add_argument("--llm-sweep-tasks-per-cap", type=int, default=48)
    release.add_argument(
        "--llm-budget-caps",
        default="2,4,6,10",
        help="Comma-separated enforced tool-call caps for the LLM budget sweep.",
    )
    release.add_argument("--max-llm-minutes", type=float, default=180.0)
    release.add_argument("--llm-max-tokens", type=int, default=128)
    release.add_argument("--llm-progress-every", type=int, default=5)
    release.add_argument("--llm-thinking", action="store_true", help="Allow Qwen reasoning tokens; slower on local hardware.")
    release.add_argument("--force-llm-panel", action="store_true")

    status = subparsers.add_parser("status-v2-release", help="Show current v2 release-run status")
    status.add_argument("--out", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        split_overrides = None
        if args.train_size is not None or args.dev_size is not None or args.test_size is not None:
            if args.train_size is None or args.dev_size is None or args.test_size is None:
                print("generate_error: set all of --train-size, --dev-size, and --test-size together.")
                return 2
            split_overrides = {
                "train": args.train_size,
                "dev": args.dev_size,
                "test_public": args.test_size,
            }

        if args.version in {"v0.2", "v2", "version2"}:
            payload = generate_dataset_v2(
                version="v0.2",
                out_dir=args.out,
                seed=args.seed,
                split_sizes=split_overrides,
                size_profile=args.size_profile,
            )
        else:
            payload = generate_dataset(
                version=args.version,
                out_dir=args.out,
                seed=args.seed,
                write_parquet=not args.no_parquet,
                split_sizes=split_overrides,
                size_profile=args.size_profile,
            )
        print(
            f"generated version={args.version} out={payload['out_dir']} "
            f"splits={payload['splits']} profile={payload['size_profile']} "
            f"quality_issues={payload['quality_issues']} manifest={payload['manifest']}"
        )
        return 0

    if args.command == "eval":
        try:
            tasks = load_tasks(args.dataset, args.split)
            agent = build_runtime_agent(
                agent_name=args.agent,
                agent_module=args.agent_module,
                agent_kwargs_raw=args.agent_kwargs,
            )
            episodes = [run_episode(task, agent) for task in tasks]
            scored = score_episodes(episodes)
            traces = [episode_to_trace_row(result) for result in episodes]
            paths = write_evaluation_artifacts(args.report, scored, traces)
            print(
                f"evaluated={len(episodes)} report={paths['report']} "
                f"traces={paths['traces']}"
            )
            return 0
        except (FileNotFoundError, AgentLoadError, ValueError, json.JSONDecodeError) as exc:
            print(f"eval_error: {exc}")
            return 2

    if args.command == "reproduce-paper":
        config = load_experiment_config(args.config)
        payload = run_experiment_pipeline(config)
        print(
            f"reproduced report={payload['report']} "
            f"tables={json.dumps(payload['tables'], sort_keys=True)} "
            f"plots={json.dumps(payload['plots'], sort_keys=True)}"
        )
        return 0

    if args.command == "run-v2-release":
        try:
            llm_budget_caps = tuple(int(value.strip()) for value in args.llm_budget_caps.split(",") if value.strip())
            payload = run_v2_release(
                out_dir=args.out,
                dataset_dir=args.dataset,
                seed=args.seed,
                llm_base_url=args.llm_base_url,
                llm_model=args.llm_model,
                calibration_tasks=args.calibration_tasks,
                llm_panel_tasks=args.llm_panel_tasks,
                llm_sweep_tasks_per_cap=args.llm_sweep_tasks_per_cap,
                llm_budget_caps=llm_budget_caps,
                llm_max_tokens=args.llm_max_tokens,
                llm_disable_thinking=not args.llm_thinking,
                llm_progress_every=args.llm_progress_every,
                max_llm_minutes=args.max_llm_minutes,
                force_llm_panel=args.force_llm_panel,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"release_run_error: {exc}")
            return 2

    if args.command == "status-v2-release":
        try:
            print(json.dumps(read_release_status(args.out), indent=2, sort_keys=True))
            return 0
        except FileNotFoundError as exc:
            print(f"release_status_error: {exc}")
            return 2

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
