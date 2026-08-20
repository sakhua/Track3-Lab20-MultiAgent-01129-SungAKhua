"""Researcher agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

SYSTEM_PROMPT = (
    "You are a research specialist. Given the retrieved sources, write concise research "
    "notes: the key facts and claims relevant to the query, each attributed to its source "
    "title. Do not analyze or draw conclusions yet — just collect and summarize evidence."
)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self, llm_client: LLMClient | None = None, search_client: SearchClient | None = None
    ) -> None:
        self._llm_client = llm_client or LLMClient()
        self._search_client = search_client or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""

        sources = self._search_client.search(
            state.request.query, max_results=state.request.max_sources
        )
        state.sources = sources

        sources_block = "\n\n".join(
            f"Title: {source.title}\nURL: {source.url or 'n/a'}\nSnippet: {source.snippet}"
            for source in sources
        )
        user_prompt = f"Query: {state.request.query}\n\nSources:\n{sources_block}"
        response = self._llm_client.complete(SYSTEM_PROMPT, user_prompt)

        state.research_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                    "source_count": len(sources),
                },
            )
        )
        state.add_trace_event("researcher_run", {"source_count": len(sources)})
        return state
