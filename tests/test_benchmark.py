from unittest.mock import MagicMock

from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    ResearchQuery,
    SourceDocument,
)
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.services.judge_client import JudgeVerdict


def _successful_runner(query: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    state.sources = [SourceDocument(title="Source A", snippet="s")]
    state.final_answer = "Answer citing Source A."
    state.agent_results = [
        AgentResult(agent=AgentName.WRITER, content="x", metadata={"cost_usd": 0.001})
    ]
    return state


def _failing_runner(query: str) -> ResearchState:
    raise RuntimeError("boom")


def _incomplete_runner(query: str) -> ResearchState:
    return ResearchState(request=ResearchQuery(query=query))


def test_run_benchmark_measures_success_metrics() -> None:
    state, metrics = run_benchmark("test-run", "Explain multi-agent systems", _successful_runner)

    assert state.final_answer is not None
    assert metrics.run_name == "test-run"
    assert metrics.latency_seconds >= 0
    assert metrics.estimated_cost_usd == 0.001
    assert metrics.citation_coverage == 1.0
    assert metrics.failure_rate == 0.0


def test_run_benchmark_captures_runner_exception() -> None:
    _, metrics = run_benchmark("failing-run", "Explain multi-agent systems", _failing_runner)

    assert metrics.failure_rate == 1.0
    assert "boom" in metrics.notes


def test_run_benchmark_flags_missing_final_answer() -> None:
    _, metrics = run_benchmark("incomplete-run", "Explain multi-agent systems", _incomplete_runner)

    assert metrics.failure_rate == 1.0
    assert "final_answer" in metrics.notes


def test_run_benchmark_uses_judge_client_when_provided() -> None:
    mock_judge = MagicMock()
    mock_judge.score.return_value = JudgeVerdict(score=8.5, rationale="Good coverage.")

    _, metrics = run_benchmark(
        "test-run",
        "Explain multi-agent systems",
        _successful_runner,
        judge_client=mock_judge,
        coverage_points=["point 1"],
    )

    assert metrics.quality_score == 8.5
    mock_judge.score.assert_called_once()


def test_run_benchmark_quality_score_none_without_judge() -> None:
    _, metrics = run_benchmark("test-run", "Explain multi-agent systems", _successful_runner)
    assert metrics.quality_score is None
