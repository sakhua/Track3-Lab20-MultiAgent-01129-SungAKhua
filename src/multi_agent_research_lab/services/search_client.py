"""Search client backed by the offline research corpus.

No internet access: this reads one topic JSON from
`data/ai_agent_offline_research_corpus_v2/topics/` and filters its embedded
source documents and facts by keyword overlap with the query.
"""

import json
import re
from pathlib import Path
from typing import Any

from multi_agent_research_lab.core.schemas import SourceDocument

_DEFAULT_CORPUS_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "ai_agent_offline_research_corpus_v2"
    / "topics"
    / "01_single_agent_vs_multi_agent_architectures_for_complex_research_tasks.json"
)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "of", "on", "or", "the", "to", "what", "when", "which", "with",
}  # fmt: skip


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {word for word in words if word not in _STOPWORDS and len(word) > 2}


class SearchClient:
    """Reads and ranks entries from a single offline knowledge-corpus topic file."""

    def __init__(self, corpus_path: Path = _DEFAULT_CORPUS_PATH) -> None:
        self._corpus_path = corpus_path
        self._topic: dict[str, Any] = json.loads(corpus_path.read_text(encoding="utf-8"))

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Rank embedded source documents and facts by keyword overlap with the query."""

        query_keywords = _keywords(query)
        knowledge_base = self._topic["knowledge_base"]

        candidates: list[tuple[int, SourceDocument]] = []
        for doc in knowledge_base["source_documents"]:
            haystack = f"{doc['title']} {doc['full_text']}"
            score = len(query_keywords & _keywords(haystack))
            snippet = doc["full_text"][:600]
            candidates.append(
                (
                    score,
                    SourceDocument(
                        title=doc["title"],
                        url=doc.get("provenance_url"),
                        snippet=snippet,
                        metadata={
                            "source_id": doc["document_id"],
                            "document_class": doc["document_class"],
                            "is_synthetic": doc["is_synthetic"],
                        },
                    ),
                )
            )

        candidates.sort(key=lambda pair: pair[0], reverse=True)
        top = [doc for score, doc in candidates if score > 0][:max_results]
        if not top:
            top = [doc for _, doc in candidates[:max_results]]
        return top

    def gold_coverage_points(self) -> list[str]:
        """Rubric coverage points the benchmark's `expected_report` calls for."""

        expected_report = self._topic["research_task"]["expected_report"]
        return list(expected_report.get("gold_coverage_points", []))
