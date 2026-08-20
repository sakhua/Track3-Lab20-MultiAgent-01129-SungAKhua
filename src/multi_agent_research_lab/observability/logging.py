"""Logging and tracing-provider setup."""

import logging
import os

from multi_agent_research_lab.core.config import Settings


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def configure_tracing(settings: Settings) -> None:
    """Enable LangSmith tracing for LangGraph if a key is configured.

    LangGraph/LangChain pick up tracing automatically from these env vars;
    no explicit client wiring is needed in the workflow code.
    """

    if not settings.langsmith_api_key:
        return
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
