from unittest.mock import MagicMock, patch

import pytest

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.services.llm_client import LLMClient


def _settings_with_key() -> Settings:
    return Settings(openai_api_key="test-key", openai_model="gpt-4o-mini")


def test_missing_api_key_raises() -> None:
    with pytest.raises(AgentExecutionError):
        LLMClient(settings=Settings(openai_api_key=None))


@patch("multi_agent_research_lab.services.llm_client.OpenAI")
def test_complete_returns_response_with_cost(mock_openai_cls: MagicMock) -> None:
    mock_client = mock_openai_cls.return_value
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="hello world"))]
    mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
    mock_client.chat.completions.create.return_value = mock_response

    client = LLMClient(settings=_settings_with_key())
    result = client.complete("system", "user")

    assert result.content == "hello world"
    assert result.input_tokens == 100
    assert result.output_tokens == 50
    assert result.cost_usd is not None
    assert result.cost_usd > 0


@patch("multi_agent_research_lab.services.llm_client.OpenAI")
def test_complete_wraps_provider_errors(mock_openai_cls: MagicMock) -> None:
    mock_client = mock_openai_cls.return_value
    mock_client.chat.completions.create.side_effect = RuntimeError("boom")

    client = LLMClient(settings=_settings_with_key())
    with pytest.raises(AgentExecutionError):
        client.complete("system", "user")
