"""Benchmark harness for single-agent vs multi-agent runs."""

import logging
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.agents.critic import citation_coverage
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.judge_client import JudgeClient

Runner = Callable[[str], ResearchState]

logger = logging.getLogger(__name__)


def _estimated_cost_usd(state: ResearchState) -> float | None:
    costs = [
        result.metadata["cost_usd"]
        for result in state.agent_results
        if result.metadata.get("cost_usd") is not None
    ]
    return sum(costs) if costs else None


def _quality_score(
    judge_client: JudgeClient | None,
    query: str,
    state: ResearchState,
    coverage_points: list[str] | None,
) -> float | None:
    if judge_client is None or not state.final_answer:
        return None
    try:
        verdict = judge_client.score(query, state.final_answer, coverage_points)
    except AgentExecutionError as exc:
        logger.warning("LLM-judge scoring failed, leaving quality_score empty: %s", exc)
        return None
    return verdict.score


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
    judge_client: JudgeClient | None = None,
    coverage_points: list[str] | None = None,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run `runner(query)` once and measure latency, cost, citation coverage, and failures.

    If `judge_client` is provided, also scores the final answer 0-10 with an
    independent LLM judge against `coverage_points`.
    """

    started = perf_counter()
    failure_rate = 0.0
    notes = ""
    try:
        state = runner(query)
    except Exception as exc:  # noqa: BLE001 - a failed run is a valid benchmark outcome
        latency = perf_counter() - started
        empty_state = ResearchState(request=ResearchQuery(query=query))
        metrics = BenchmarkMetrics(
            run_name=run_name,
            latency_seconds=latency,
            failure_rate=1.0,
            notes=f"runner raised {type(exc).__name__}: {exc}",
        )
        return empty_state, metrics
    latency = perf_counter() - started

    if not state.final_answer:
        failure_rate = 1.0
        notes = "runner completed without producing a final_answer"

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=_estimated_cost_usd(state),
        quality_score=_quality_score(judge_client, query, state, coverage_points),
        citation_coverage=citation_coverage(state) if state.sources else None,
        failure_rate=failure_rate,
        notes=notes,
    )
    return state, metrics
