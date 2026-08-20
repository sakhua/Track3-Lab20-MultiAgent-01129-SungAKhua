"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def _fmt(value: float | None, suffix: str = "", precision: int = 4) -> str:
    if value is None:
        return ""
    if suffix == "%":
        return f"{value:.0%}"
    return f"{value:.{precision}f}{suffix}"


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to a markdown table plus a short comparison."""

    lines = [
        "# Benchmark Report",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} "
            f"| {_fmt(item.estimated_cost_usd)} | {_fmt(item.quality_score, precision=1)} "
            f"| {_fmt(item.citation_coverage, '%')} | {_fmt(item.failure_rate, '%')} "
            f"| {item.notes} |"
        )
    lines.append("")

    if len(metrics) >= 2:
        fastest = min(metrics, key=lambda m: m.latency_seconds)
        cost_candidates = [m for m in metrics if m.estimated_cost_usd is not None]
        cheapest = (
            min(cost_candidates, key=lambda m: m.estimated_cost_usd or 0.0)
            if cost_candidates
            else None
        )
        lines.append("## Comparison")
        lines.append("")
        lines.append(f"- Fastest run: **{fastest.run_name}** ({fastest.latency_seconds:.2f}s)")
        if cheapest is not None:
            lines.append(
                f"- Cheapest run: **{cheapest.run_name}** (${cheapest.estimated_cost_usd:.4f})"
            )
        lines.append("")

    return "\n".join(lines) + "\n"
