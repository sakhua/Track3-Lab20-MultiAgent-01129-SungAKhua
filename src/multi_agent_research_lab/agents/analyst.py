"""Analyst agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

SYSTEM_PROMPT = (
    "You are a research analyst. Given research notes and their sources, extract the key "
    "claims, compare viewpoints, flag weak or unsupported evidence, and note any "
    "disagreements between sources. Distinguish real public references from synthetic or "
    "benchmark evidence when the source metadata indicates it. Do not write the final "
    "answer yet — produce structured analysis notes."
)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""

        sources_block = "\n".join(
            f"- {source.title} (synthetic={source.metadata.get('is_synthetic', 'unknown')})"
            for source in state.sources
        )
        user_prompt = (
            f"Query: {state.request.query}\n\n"
            f"Research notes:\n{state.research_notes}\n\n"
            f"Sources used:\n{sources_block}"
        )
        response = self._llm_client.complete(SYSTEM_PROMPT, user_prompt)

        state.analysis_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event("analyst_run", {})
        return state
