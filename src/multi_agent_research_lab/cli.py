"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError, StudentTodoError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.judge_client import JudgeClient
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

BASELINE_SYSTEM_PROMPT = (
    "You are a single research assistant. Research the user's query, analyze the "
    "findings, and write a clear, well-structured answer for a technical audience, "
    "all in one pass without delegating to other agents."
)


def _run_baseline(query: str) -> ResearchState:
    """Single LLM call does research + analysis + writing in one pass."""

    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    response = LLMClient().complete(BASELINE_SYSTEM_PROMPT, request.query)
    state.final_answer = response.content
    state.agent_results.append(
        AgentResult(
            agent=AgentName.WRITER,
            content=response.content,
            metadata={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )
    )
    return state


def _run_multi_agent(query: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    return MultiAgentWorkflow().run(state)


app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline: one LLM call does research + analysis + writing."""

    _init()
    _parse_query(query)  # validate early for a clean error message
    try:
        state = _run_baseline(query)
    except AgentExecutionError as exc:
        console.print(Panel.fit(str(exc), title="LLM Error", style="red"))
        raise typer.Exit(code=1) from exc
    result = state.agent_results[-1]
    cost = result.metadata.get("cost_usd")
    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))
    console.print(
        f"[dim]tokens in={result.metadata.get('input_tokens')} "
        f"out={result.metadata.get('output_tokens')} cost=${cost:.6f}[/dim]"
        if cost is not None
        else "[dim]token/cost metadata unavailable[/dim]"
    )


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow: Supervisor -> Researcher -> Analyst -> Writer."""

    _init()
    _parse_query(query)  # validate early for a clean error message
    try:
        result = _run_multi_agent(query)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    console.print(result.model_dump_json(indent=2))


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run baseline and multi-agent on the same query and write a comparison report."""

    _init()
    _parse_query(query)  # validate early for a clean error message

    settings = get_settings()
    judge_client: JudgeClient | None = None
    if settings.gemini_api_key:
        judge_client = JudgeClient(settings)
    else:
        console.print(
            "[dim]GEMINI_API_KEY not set - skipping LLM-judge quality scoring.[/dim]"
        )
    coverage_points = SearchClient().gold_coverage_points()

    _, baseline_metrics = run_benchmark(
        "single-agent baseline", query, _run_baseline, judge_client, coverage_points
    )
    _, multi_metrics = run_benchmark(
        "multi-agent", query, _run_multi_agent, judge_client, coverage_points
    )

    report = render_markdown_report([baseline_metrics, multi_metrics])
    path = LocalArtifactStore().write_text("benchmark_report.md", report)

    console.print(report)
    console.print(Panel.fit(f"Report written to {path}", title="Benchmark", style="green"))


if __name__ == "__main__":
    app()
