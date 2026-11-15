"""Backend-owned injection boundaries for search, documents and LLM.

The search_adapter bridge translates these calls to search_service.SearchService.

Integration requirements:
- SearchFactory(archive_path: Path, settings: Settings) initializes once and
  returns SearchComponents(evidence, documents). solution.run already does this;
  API bootstrap must do so once before create_app, never once per question.
- Assistant prepares SearchQuery before calling EvidenceProvider. The adapter
  translates this payload to the search service API; it never
  receives ChatRequest or assistant dialogue.
- Query fields: current query, bounded prior USER context, confirmed filters.
  No automatic date/document filters are inferred yet. Context is not evidence.
- Evidence returns list[Fragment], ranked most useful first; [] is a successful
  empty search. Fragment IDs must be unique, stable and refer to real sources.
- Documents returns Document for a stable exact revision ID, None for unknown ID.
- Raise BackendUnavailable for absent setup, UpstreamError for service/data
  failures, UpstreamTimeout for deadlines. Do not turn failures into [] or None.
- Providers share one read-only index. The adapter runs synchronous search in
  a worker thread, so it does not block the async event loop.
"""
from __future__ import annotations

from typing import Protocol
from schemas import ChatRequest, Document, Fragment


class EvidenceProvider(Protocol):
    async def __call__(self, request: SearchQuery) -> list[Fragment]: ...


class DocumentProvider(Protocol):
    async def __call__(self, document_id: str) -> Document | None: ...


class LanguageModel(Protocol):
    async def complete(self, messages: list[dict[str, str]]) -> str: ...


class BackendUnavailable(Exception):
    """Missing configuration or dependency."""


class UpstreamError(Exception):
    """Failed or malformed upstream response."""


class UpstreamTimeout(Exception):
    """Upstream deadline exceeded."""


# Backend integration contract; the search module has its own public API.
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config import Settings


@dataclass(frozen=True)
class SearchComponents:
    evidence: EvidenceProvider
    documents: DocumentProvider


class SearchFactory(Protocol):
    async def __call__(self, archive_path: Path, settings: "Settings") -> SearchComponents:
        """Initialize once; return reusable, concurrency-safe provider callbacks."""
        ...


async def initialize_search(archive_path: Path, settings: "Settings") -> SearchComponents:
    """Build one reusable corpus/index and return the backend providers."""
    from search_adapter import create_search_components
    return await create_search_components(archive_path, settings)


@dataclass(frozen=True)
class SearchQuery:
    """Proposed bridge payload, NOT participant #1's existing API.

    Context is untrusted user input, never legal evidence. Filters must contain
    only confirmed constraints; the default bridge makes no guesses about dates
    or document identities.
    """
    query: str
    user_context: tuple[str, ...] = ()
    filters: tuple[tuple[str, str], ...] = ()


def prepare_search_query(request: ChatRequest) -> SearchQuery:
    # Revalidate to apply the same budget even to mutated injected requests.
    request = ChatRequest.model_validate(request.model_dump())
    context = tuple(m.content for m in request.history if m.role == "user")
    return SearchQuery(query=request.question, user_context=context)
