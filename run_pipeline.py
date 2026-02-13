"""
sust-AI-naible — Full Pipeline Runner
======================================
Runs the multi-agent system via the Orchestrator.

The Orchestrator manages 5 AI agents + 2 deterministic services:
  - Planner Agent       (LLM reasoning + deterministic solver)
  - Governance Agent    (LLM risk assessment + deterministic rules)
  - Executor Agent      (LLM ticket generation + deterministic execution)
  - Developer Copilot   (LLM summaries + deterministic points)
  - Verifier            (deterministic counterfactual verification)
  - Carbon Accountant   (deterministic emissions math)
  - Ingestor            (simulated data generation)

Usage:
  python run_pipeline.py              # Mock LLM (no API key needed)
  OPENAI_API_KEY=sk-... python run_pipeline.py   # Real LLM
"""

from datetime import datetime
from src.orchestrator import Orchestrator


def main():
    orchestrator = Orchestrator(
        llm_provider="auto",  # Uses OpenAI if OPENAI_API_KEY is set, else mock
        verbose=True,
    )

    summary = orchestrator.run(
        sim_start=datetime(2025, 1, 1),
        sim_days=30,
        seed=42,
        time_resolution_hours=4,
    )

    # Print which LLM was used
    print(f"\n  LLM Provider: {summary.get('llm_provider', 'unknown')}")
    print(f"  Agent reasoning steps:")
    for agent, stats in summary.get("agents", {}).items():
        print(f"    {agent}: {stats['reasoning_steps']} reasoning steps, "
              f"{stats['actions_taken']} tool calls")


if __name__ == "__main__":
    main()
