"""Reference baseline agents."""

from toolmisusebench.baselines.heuristic_agent import HeuristicAgent
from toolmisusebench.baselines.local_llm_adapter import LocalLLMAdapter
from toolmisusebench.baselines.policy_aware_agent import PolicyAwareAgent
from toolmisusebench.baselines.schema_repair_agent import SchemaRepairAgent
from toolmisusebench.version2.baselines import InstructionParserAgent, InstructionParserNoRecoveryAgent


def build_agent(agent_name: str):
    if agent_name == "heuristic":
        return HeuristicAgent()
    if agent_name == "schema_repair":
        return SchemaRepairAgent()
    if agent_name == "policy_aware":
        return PolicyAwareAgent()
    if agent_name == "local_llm":
        return LocalLLMAdapter()
    if agent_name in {"instruction_parser", "instruction_parser_v2"}:
        return InstructionParserAgent()
    if agent_name in {"instruction_parser_no_recovery", "instruction_parser_no_recovery_v2"}:
        return InstructionParserNoRecoveryAgent()
    raise ValueError(
        "Unsupported agent "
        f"'{agent_name}'. Available: heuristic, schema_repair, policy_aware, local_llm, instruction_parser, instruction_parser_no_recovery"
    )


__all__ = [
    "HeuristicAgent",
    "SchemaRepairAgent",
    "PolicyAwareAgent",
    "LocalLLMAdapter",
    "InstructionParserAgent",
    "InstructionParserNoRecoveryAgent",
    "build_agent",
]
