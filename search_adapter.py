"""Bridge the root-level search service to backend evidence/document protocols."""

from __future__ import annotations

import asyncio
import zipfile
from pathlib import Path

from config import Settings
from dependencies import (BackendUnavailable, SearchComponents, SearchQuery,
                          UpstreamError, UpstreamTimeout)
from schemas import Document, Fragment
from search_service import SearchService, load_search_service


def bind_search_service(service: SearchService, settings: Settings) -> SearchComponents:
    """Bind one reusable index to async backend callbacks."""

    async def evidence(request: SearchQuery) -> list[Fragment]:
        try:
            hits = await asyncio.to_thread(
                service.search, request.query, context=request.user_context,
                filters=request.filters, top_k=settings.top_k,
                context_budget_chars=settings.max_context_chars,
            )
            results = []
            for hit in hits:
                item = hit.fragment
                source = service.get_document(item.document_id)
                if source is None or source.text[item.start:item.end] != item.text:
                    raise UpstreamError("Search result does not match its source document")
                results.append(Fragment(
                    fragment_id=item.fragment_id,
                    document_id=item.document_id,
                    document_title=item.document_title,
                    section=item.section,
                    text=item.text,
                    revision=item.revision,
                    valid_from=item.valid_from,
                    valid_to=item.valid_to,
                    start=None,
                    end=None,
                ))
            return results
        except TimeoutError as exc:
            raise UpstreamTimeout("Search timed out") from exc
        except ValueError as exc:
            raise UpstreamError("Search request or result is invalid") from exc

    async def documents(document_id: str) -> Document | None:
        try:
            source = service.get_document(document_id)
            if source is None:
                return None
            return Document(document_id=source.document_id, text=source.text)
        except TimeoutError as exc:
            raise UpstreamTimeout("Document lookup timed out") from exc
        except ValueError as exc:
            raise UpstreamError("Document result is invalid") from exc

    return SearchComponents(evidence=evidence, documents=documents)


async def create_search_components(archive_path: Path, settings: Settings) -> SearchComponents:
    """Load the archive and index once without blocking the async event loop."""
    try:
        service = await asyncio.to_thread(load_search_service, archive_path)
    except FileNotFoundError as exc:
        raise BackendUnavailable("Knowledge base archive is missing") from exc
    except TimeoutError as exc:
        raise UpstreamTimeout("Knowledge base indexing timed out") from exc
    except (OSError, zipfile.BadZipFile, ValueError) as exc:
        raise UpstreamError("Knowledge base archive could not be indexed") from exc
    return bind_search_service(service, settings)
