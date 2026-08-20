"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass

from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError

# USD per 1M tokens, gpt-4o-mini pricing. Approximate — good enough for relative benchmark cost.
_PRICING_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
}
_DEFAULT_PRICING = (0.15, 0.60)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


def _estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    input_price, output_price = _PRICING_PER_1M_TOKENS.get(model, _DEFAULT_PRICING)
    return (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price


class LLMClient:
    """OpenAI-backed LLM client.

    Retry/timeout/token accounting lives here so agents stay provider-agnostic.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        if not self._settings.openai_api_key:
            raise AgentExecutionError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        self._client = OpenAI(
            api_key=self._settings.openai_api_key,
            timeout=self._settings.timeout_seconds,
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion from OpenAI's chat completions API."""

        try:
            response = self._client.chat.completions.create(
                model=self._settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a domain error after retries
            raise AgentExecutionError(f"LLM completion failed: {exc}") from exc

        content = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None
        cost_usd = (
            _estimate_cost_usd(self._settings.openai_model, input_tokens, output_tokens)
            if input_tokens is not None and output_tokens is not None
            else None
        )
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
