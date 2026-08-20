from unittest.mock import MagicMock

from multi_agent_research_lab.agents.critic import CriticAgent, citation_coverage
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMResponse


def _state_with_sources(final_answer: str | None) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.sources = [
        SourceDocument(title="AutoGen", snippet="s"),
        SourceDocument(title="MetaGPT", snippet="s"),
    ]
    state.final_answer = final_answer
    return state


def test_citation_coverage_counts_titles_mentioned_in_final_answer() -> None:
    state = _state_with_sources("This answer cites AutoGen but not the other source.")
    assert citation_coverage(state) == 0.5


def test_citation_coverage_zero_without_final_answer() -> None:
    state = _state_with_sources(None)
    assert citation_coverage(state) == 0.0


def test_critic_run_appends_error_when_issues_found() -> None:
    mock_llm = MagicMock()
    mock_llm.complete.return_value = LLMResponse(content="- Unsupported claim about X")
    state = _state_with_sources("Answer citing AutoGen and MetaGPT.")

    result = CriticAgent(llm_client=mock_llm).run(state)

    assert any("critic_findings" in error for error in result.errors)
    assert result.trace[-1]["name"] == "critic_run"


def test_critic_run_no_error_when_clean() -> None:
    mock_llm = MagicMock()
    mock_llm.complete.return_value = LLMResponse(content="No issues found.")
    state = _state_with_sources("Answer citing AutoGen and MetaGPT.")

    result = CriticAgent(llm_client=mock_llm).run(state)

    assert result.errors == []
