from __future__ import annotations

import json

from toolmisusebench.cli.main import main
from toolmisusebench.types import Observation, RemainingBudget
from toolmisusebench.version2.generator import generate_dataset_v2
from toolmisusebench.version2.llm_agent import OpenAICompatibleToolAgent
from toolmisusebench.version2.release_runner import read_release_status, run_v2_release


def test_release_runner_records_resumable_deterministic_stages(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    out = tmp_path / "release"
    generate_dataset_v2(
        version="v0.2",
        out_dir=dataset,
        seed=3,
        split_sizes={"train": 12, "dev": 12, "test_public": 12},
    )

    summary = run_v2_release(
        out_dir=out,
        dataset_dir=dataset,
        seed=3,
        llm_base_url=None,
        llm_model="unused",
    )

    assert summary["status"]["state"] == "complete"
    status = read_release_status(out)
    assert status["stages"]["dataset"]["state"] == "skipped"
    assert status["stages"]["deterministic_instruction_parser"]["state"] == "complete"
    assert status["stages"]["llm_calibration"]["state"] == "skipped"
    assert (out / "run_events.jsonl").exists()
    assert (out / "dataset_quality_report.json").exists()
    assert (out / "fault_ablation_summary.json").exists()
    assert (out / "budget_sweep_deterministic.json").exists()
    assert (out / "stages" / "deterministic_heuristic" / "episodes.jsonl").exists()
    assert (out / "stages" / "budget_sweep_instruction_parser_cap_2" / "score.json").exists()

    second = run_v2_release(
        out_dir=out,
        dataset_dir=dataset,
        seed=3,
        llm_base_url=None,
        llm_model="unused",
    )
    assert second["status"]["state"] == "complete"
    assert second["status"]["stages"]["deterministic_heuristic"]["state"] == "skipped"


def test_release_status_cli(tmp_path, capsys) -> None:
    out = tmp_path / "release"
    out.mkdir()
    (out / "run_status.json").write_text(json.dumps({"state": "running", "stages": {}}), encoding="utf-8")

    code = main(["status-v2-release", "--out", str(out)])
    assert code == 0
    assert '"state": "running"' in capsys.readouterr().out


def test_openai_compatible_agent_parses_native_tool_call() -> None:
    agent = OpenAICompatibleToolAgent(base_url="http://example.invalid/v1", model="test")
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_doc",
                                "arguments": '{"doc_id":"d1"}',
                            }
                        }
                    ]
                }
            }
        ]
    }
    action = agent._parse_action(response)
    assert action is not None
    assert action.tool_name == "get_doc"
    assert action.args == {"doc_id": "d1"}


def test_openai_compatible_agent_uses_compact_tool_call_prompt() -> None:
    agent = OpenAICompatibleToolAgent(base_url="http://127.0.0.1:8080/v1", model="test")
    observation = Observation(
        instruction="Fetch d1",
        tool_schemas=[
            {
                "name": "get_doc",
                "properties": {"doc_id": {"type": "string"}},
                "required": ["doc_id"],
                "additionalProperties": False,
            }
        ],
        transcript=[],
        remaining_budget=RemainingBudget(steps_left=3, tool_calls_left=3, retries_left=1),
        last_error=None,
    )

    assert "/no_think" in agent._messages(observation)[0]["content"]
    first_tools = agent._tools(observation)
    assert agent._tools(observation) is first_tools
