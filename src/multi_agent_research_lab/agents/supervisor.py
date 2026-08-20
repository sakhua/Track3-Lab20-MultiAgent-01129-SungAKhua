"""Supervisor / router."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState

ROUTE_RESEARCHER = "researcher"
ROUTE_ANALYST = "analyst"
ROUTE_WRITER = "writer"
ROUTE_DONE = "done"


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def decide_route(self, state: ResearchState) -> str:
        """Pick the next route based on what's missing in shared state."""

        if state.iteration >= get_settings().max_iterations:
            return ROUTE_DONE
        if not state.sources or not state.research_notes:
            return ROUTE_RESEARCHER
        if not state.analysis_notes:
            return ROUTE_ANALYST
        if not state.final_answer:
            return ROUTE_WRITER
        return ROUTE_DONE

    def run(self, state: ResearchState) -> ResearchState:
        """Record the next route in `state.route_history`."""

        route = self.decide_route(state)
        state.record_route(route)
        state.add_trace_event("supervisor_route", {"route": route, "iteration": state.iteration})
        return state
