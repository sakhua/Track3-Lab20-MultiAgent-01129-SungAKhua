"""LangGraph workflow wiring supervisor + worker agents."""

from typing import Any

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import (
    ROUTE_ANALYST,
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
    SupervisorAgent,
)
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span


def _traced_node(agent: BaseAgent) -> Any:
    def node(state: ResearchState) -> ResearchState:
        with trace_span(f"{agent.name}_span", {"iteration": state.iteration}):
            return agent.run(state)

    return node


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(
        self,
        supervisor: SupervisorAgent | None = None,
        researcher: ResearcherAgent | None = None,
        analyst: AnalystAgent | None = None,
        writer: WriterAgent | None = None,
    ) -> None:
        self._supervisor = supervisor or SupervisorAgent()
        self._researcher = researcher or ResearcherAgent()
        self._analyst = analyst or AnalystAgent()
        self._writer = writer or WriterAgent()
        self._graph = self.build()

    def build(self) -> Any:
        """Create the LangGraph graph: supervisor routes to each worker in turn."""

        graph = StateGraph(ResearchState)
        graph.add_node("supervisor", _traced_node(self._supervisor))
        graph.add_node(ROUTE_RESEARCHER, _traced_node(self._researcher))
        graph.add_node(ROUTE_ANALYST, _traced_node(self._analyst))
        graph.add_node(ROUTE_WRITER, _traced_node(self._writer))

        graph.set_entry_point("supervisor")
        graph.add_conditional_edges(
            "supervisor",
            lambda state: state.route_history[-1],
            {
                ROUTE_RESEARCHER: ROUTE_RESEARCHER,
                ROUTE_ANALYST: ROUTE_ANALYST,
                ROUTE_WRITER: ROUTE_WRITER,
                ROUTE_DONE: END,
            },
        )
        graph.add_edge(ROUTE_RESEARCHER, "supervisor")
        graph.add_edge(ROUTE_ANALYST, "supervisor")
        graph.add_edge(ROUTE_WRITER, "supervisor")

        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return the final state."""

        result = self._graph.invoke(state)
        return ResearchState.model_validate(result)
