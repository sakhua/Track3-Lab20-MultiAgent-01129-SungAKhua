"""Optional critic agent for bonus work: citation coverage + LLM fact-check."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

SYSTEM_PROMPT = (
    "You are a fact-checking critic. Given a final answer and the research/analysis notes "
    "it should be grounded in, list any claims in the final answer that are NOT supported "
    "by the notes (potential hallucinations), and any major claims from the notes that were "
    "dropped. Be terse: bullet points only, no preamble. If nothing is wrong, say 'No issues "
    "found.'"
)


def citation_coverage(state: ResearchState) -> float:
    """Fraction of sources whose title is mentioned in the final answer."""

    if not state.sources or not state.final_answer:
        return 0.0
    cited = sum(1 for source in state.sources if source.title in state.final_answer)
    return cited / len(state.sources)


class CriticAgent(BaseAgent):
    """Fact-checking and citation-coverage review agent."""

    name = "critic"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate `state.final_answer` and append findings to `state.errors`."""

        coverage = citation_coverage(state)
        user_prompt = (
            f"Research notes:\n{state.research_notes}\n\n"
            f"Analysis notes:\n{state.analysis_notes}\n\n"
            f"Final answer:\n{state.final_answer}"
        )
        response = self._llm_client.complete(SYSTEM_PROMPT, user_prompt)

        if "no issues found" not in response.content.strip().lower():
            state.errors.append(f"critic_findings: {response.content}")
        state.add_trace_event(
            "critic_run",
            {"citation_coverage": coverage, "findings": response.content},
        )
        return state
