"""LLM-as-judge quality scoring, backed by a different provider (Gemini) than
the agents (OpenAI). Using an independent model avoids a system scoring its own
output favorably.
"""

import json
import re
from dataclasses import dataclass

from google import genai
from google.genai import types

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError

SYSTEM_PROMPT = (
    "You are an impartial research-report grader. Score the given answer from 0 to 10 "
    "against the research query and the listed coverage criteria. Weigh factual grounding, "
    "structure, and whether the criteria are substantively addressed (not just mentioned). "
    "Respond with strict JSON only: "
    '{"score": <float 0-10>, "rationale": "<one or two sentences>"}'
)


@dataclass(frozen=True)
class JudgeVerdict:
    score: float
    rationale: str


def _build_prompt(query: str, answer: str, coverage_points: list[str]) -> str:
    criteria = "\n".join(f"- {point}" for point in coverage_points) or "(no specific criteria)"
    return (
        f"Research query: {query}\n\n"
        f"Coverage criteria the answer should address:\n{criteria}\n\n"
        f"Answer to grade:\n{answer}"
    )


def _parse_verdict(raw_text: str) -> JudgeVerdict:
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise AgentExecutionError(f"Judge response was not JSON: {raw_text!r}")
    payload = json.loads(match.group(0))
    score = max(0.0, min(10.0, float(payload["score"])))
    return JudgeVerdict(score=score, rationale=str(payload.get("rationale", "")))


class JudgeClient:
    """Gemini-backed quality judge, independent of the OpenAI agents being scored."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        if not self._settings.gemini_api_key:
            raise AgentExecutionError(
                "GEMINI_API_KEY is not set. Fill it into .env to enable LLM-judge scoring."
            )
        self._client = genai.Client(api_key=self._settings.gemini_api_key)

    def score(
        self, query: str, answer: str, coverage_points: list[str] | None = None
    ) -> JudgeVerdict:
        """Score one answer 0-10 against the query and optional coverage criteria."""

        prompt = _build_prompt(query, answer, coverage_points or [])
        try:
            response = self._client.models.generate_content(
                model=self._settings.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a domain error
            raise AgentExecutionError(f"Judge call failed: {exc}") from exc

        text = response.text
        if not text:
            raise AgentExecutionError("Judge returned an empty response")
        return _parse_verdict(text)
