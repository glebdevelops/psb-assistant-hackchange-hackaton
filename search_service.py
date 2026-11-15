"""Public synchronous search and document API for the PSB knowledge base.

Create one service at startup. Its corpus and index are read-only afterwards,
so concurrent callers can share it; an async backend should call search via
asyncio.to_thread rather than blocking its event loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from search_corpus import Corpus, CorpusDocument, CorpusFragment, load_corpus
from search_index import SearchIndex


_FILTERS = frozenset({"document_id", "kind", "revision", "source_name"})
_DAY = re.compile(r"(?<!\d)(\d{1,2})\.(\d{1,2})\.(20\d{2})(?!\d)")
_YEAR = re.compile(r"\b(?:в|на|за)\s+(20\d{2})\s*(?:году|года|год|г\.)(?!\w)", re.I)


def _event_interval(query: str) -> tuple[date, date] | None:
    """Recognize only explicit event dates, never a year inside a contract ID."""
    match = _DAY.search(query)
    if match:
        try:
            day, month, year = map(int, match.groups())
            value = date(year, month, day)
            return value, value
        except ValueError:
            return None
    match = _YEAR.search(query)
    if match:
        year = int(match.group(1))
        return date(year, 1, 1), date(year, 12, 31)
    return None


def _revision_factor(document: CorpusDocument, event: tuple[date, date] | None) -> float:
    """Rank documented regulation periods, without declaring legal validity."""
    if event is None or "Регламент согласования договоров" not in document.title:
        return 1.0
    started = re.search(r"Действует с\s+(\d{2}\.\d{2}\.\d{4})", document.text, re.I)
    expired = re.search(r"УТРАТИЛА СИЛУ с\s+(\d{2}\.\d{2}\.\d{4})", document.text, re.I)

    def parse(match: re.Match[str] | None) -> date | None:
        if match is None:
            return None
        day, month, year = map(int, match.group(1).split("."))
        return date(year, month, day)

    start, end = parse(started), parse(expired)
    first, last = event
    if (start and last < start) or (end and first >= end):
        return 0.35
    if (start is None or first >= start) and (end is None or last < end):
        return 1.5
    return 1.0


@dataclass(frozen=True)
class SearchHit:
    fragment: CorpusFragment
    score: float


@dataclass(frozen=True)
class SearchService:
    corpus: Corpus
    index: SearchIndex

    @classmethod
    def from_archive(cls, archive_path: Path, *, chunk_chars: int = 1_800) -> SearchService:
        """Load the ZIP and build the index once, before serving questions."""
        corpus = load_corpus(archive_path, chunk_chars=chunk_chars)
        return cls(corpus, SearchIndex.build(corpus))

    def get_document(self, document_id: str) -> CorpusDocument | None:
        return self.corpus.get_document(document_id)

    def search(
        self,
        query: str,
        *,
        context: tuple[str, ...] = (),
        filters: tuple[tuple[str, str], ...] = (),
        top_k: int = 10,
    ) -> list[SearchHit]:
        """Return ranked source fragments; [] means no matching candidates.

        Context is prior *user* text and has lower weight than the current
        question. Filters must already be confirmed by the caller. Unknown
        filters fail loudly instead of being silently ignored. No date filter
        is offered because historical legal applicability is not inferred here.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if not isinstance(context, tuple) or any(not isinstance(item, str) for item in context):
            raise ValueError("context must be a tuple of strings")
        if not isinstance(filters, tuple):
            raise ValueError("filters must be a tuple of pairs")
        constraints: dict[str, str] = {}
        for item in filters:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("filters must contain (name, value) pairs")
            name, value = item
            if name not in _FILTERS or not isinstance(value, str) or not value:
                raise ValueError(f"unsupported search filter: {name!r}")
            if name in constraints and constraints[name] != value:
                raise ValueError(f"conflicting search filter: {name!r}")
            constraints[name] = value

        scored = self.index.score(query, context)
        event = _event_interval(query)
        if event is None and context:
            event = _event_interval(context[-1])
        candidates: list[tuple[int, float]] = []
        for index, score in scored.items():
            fragment = self.index.entries[index].fragment
            document = self.corpus.get_document(fragment.document_id)
            assert document is not None  # The immutable index came from this corpus.
            if any(getattr(document, name) != value for name, value in constraints.items()):
                continue
            candidates.append((index, score * _revision_factor(document, event)))
        candidates.sort(key=lambda item: (-item[1], self.index.entries[item[0]].fragment.fragment_id))

        # A long statutory article may contain many similar matches. Prefer
        # some source diversity while retaining the best passages from it.
        seen_per_document: dict[str, int] = {}
        ranked: list[SearchHit] = []
        for index, raw_score in candidates:
            fragment = self.index.entries[index].fragment
            count = seen_per_document.get(fragment.document_id, 0)
            seen_per_document[fragment.document_id] = count + 1
            ranked.append(SearchHit(fragment, raw_score / (1.0 + 0.12 * count)))
        ranked.sort(key=lambda hit: (-hit.score, hit.fragment.fragment_id))
        return ranked[:top_k]


def load_search_service(archive_path: Path, *, chunk_chars: int = 1_800) -> SearchService:
    """Convenient stable entry point for the future backend adapter."""
    return SearchService.from_archive(archive_path, chunk_chars=chunk_chars)
