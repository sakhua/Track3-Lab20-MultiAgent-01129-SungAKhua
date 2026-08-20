from unittest.mock import MagicMock, patch

import pytest

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.services.judge_client import JudgeClient


def _settings_with_key() -> Settings:
    return Settings(gemini_api_key="test-key", gemini_model="gemini-2.5-flash")


def test_missing_api_key_raises() -> None:
    with pytest.raises(AgentExecutionError):
        JudgeClient(settings=Settings(gemini_api_key=None))


@patch("multi_agent_research_lab.services.judge_client.genai.Client")
def test_score_parses_json_verdict(mock_client_cls: MagicMock) -> None:
    mock_client = mock_client_cls.return_value
    mock_response = MagicMock()
    mock_response.text = '{"score": 7.5, "rationale": "Solid but missing counterargument."}'
    mock_client.models.generate_content.return_value = mock_response

    judge = JudgeClient(settings=_settings_with_key())
    verdict = judge.score("query", "answer", ["point 1", "point 2"])

    assert verdict.score == 7.5
    assert "counterargument" in verdict.rationale


@patch("multi_agent_research_lab.services.judge_client.genai.Client")
def test_score_clamps_out_of_range_values(mock_client_cls: MagicMock) -> None:
    mock_client = mock_client_cls.return_value
    mock_response = MagicMock()
    mock_response.text = '{"score": 15, "rationale": "..."}'
    mock_client.models.generate_content.return_value = mock_response

    judge = JudgeClient(settings=_settings_with_key())
    verdict = judge.score("query", "answer")

    assert verdict.score == 10.0


@patch("multi_agent_research_lab.services.judge_client.genai.Client")
def test_score_raises_on_non_json_response(mock_client_cls: MagicMock) -> None:
    mock_client = mock_client_cls.return_value
    mock_response = MagicMock()
    mock_response.text = "not json at all"
    mock_client.models.generate_content.return_value = mock_response

    judge = JudgeClient(settings=_settings_with_key())
    with pytest.raises(AgentExecutionError):
        judge.score("query", "answer")
