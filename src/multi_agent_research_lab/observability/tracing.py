"""Tracing hooks.

This file intentionally avoids binding to one provider. Students can plug in LangSmith,
Langfuse, OpenTelemetry, or simple JSON traces. The default backend logs spans as
structured JSON via the standard logging module, which is enough to inspect a run
without any tracing provider credentials.
"""

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

logger = logging.getLogger("multi_agent_research_lab.trace")


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal span context used by the skeleton.

    Logs each span as a JSON line on exit. Swap the `logger.info` call for a
    LangSmith/Langfuse/OTel exporter to send spans to a real tracing backend.
    """

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started
        logger.info("trace_span %s", json.dumps(span, default=str))
