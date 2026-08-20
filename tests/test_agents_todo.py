"""Unit tests for SupervisorAgent routing policy.

Replaces the earlier skeleton guard test now that the supervisor is implemented.
"""

from multi_agent_research_lab.agents.supervisor import (
    ROUTE_ANALYST,
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
    SupervisorAgent,
)
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def _state(**overrides: object) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def test_routes_to_researcher_when_no_sources() -> None:
    route = SupervisorAgent().decide_route(_state())
    assert route == ROUTE_RESEARCHER


def test_routes_to_analyst_once_research_notes_exist() -> None:
    state = _state(
        sources=[SourceDocument(title="t", snippet="s")],
        research_notes="notes",
    )
    assert SupervisorAgent().decide_route(state) == ROUTE_ANALYST


def test_routes_to_writer_once_analysis_notes_exist() -> None:
    state = _state(
        sources=[SourceDocument(title="t", snippet="s")],
        research_notes="notes",
        analysis_notes="analysis",
    )
    assert SupervisorAgent().decide_route(state) == ROUTE_WRITER


def test_routes_to_done_once_final_answer_exists() -> None:
    state = _state(
        sources=[SourceDocument(title="t", snippet="s")],
        research_notes="notes",
        analysis_notes="analysis",
        final_answer="answer",
    )
    assert SupervisorAgent().decide_route(state) == ROUTE_DONE


def test_routes_to_done_when_max_iterations_reached() -> None:
    state = _state(iteration=6)
    assert SupervisorAgent().decide_route(state) == ROUTE_DONE


def test_run_appends_route_history_and_trace() -> None:
    state = _state()
    result = SupervisorAgent().run(state)
    assert result.route_history == [ROUTE_RESEARCHER]
    assert result.iteration == 1
    assert result.trace[-1]["name"] == "supervisor_route"
