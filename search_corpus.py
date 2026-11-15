"""Read the supplied ZIP and retain exact source text and citeable spans.

This module uses only the Python standard library and does not extract files.
All offsets are Python Unicode code-point indexes into CorpusDocument.text;
the backend's current Fragment schema does not accept numeric offsets yet.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping


MAX_ARCHIVE_MEMBERS = 10_000
MAX_DOCUMENT_BYTES = 32 * 1024 * 1024
MAX_CORPUS_BYTES = 256 * 1024 * 1024
DEFAULT_CHUNK_CHARS = 1_800

_SECTION = re.compile(r"^\s*(?:(?P<number>\d+(?:\.\d+)*)\.\s+|(?P<article>Статья\s+\d+(?:\.\d+)?)(?:\.|\s))", re.I)
_EDITORIAL = re.compile(
    r"^\s*(?:Федеральн(?:ым|ого|ый|ыми)\s+закон(?:ом|а|ы)?\s+от\b|"
    r"(?:Статья|Пункт|Часть|Абзац|Подпункт)\s+\S+\s+(?:изменен|изменена|изменены|утратил|утратила|дополнен|дополнена)\b|"
    r"О\s+проверке\s+конституционности\b|См\.\s+|ГАРАНТ:|Информация\s+об\s+изменениях\b)",
    re.I,
)
_INSTRUCTION = re.compile(
    r"(?:ИИ[-\s]?АССИСТЕНТ|(?:игнорируй|игнорировать)\s+все\s+(?:остальные\s+)?(?:документы|инструкции)|"
    r"ignore\s+(?:all\s+)?previous\s+instructions)",
    re.I,
)
_REVISION = re.compile(r"\bред(?:акция|\.)?\s*(?:от\s*)?(\d{4}|\d{2}\.\d{2}\.\d{4})\b", re.I)


@dataclass(frozen=True)
class CorpusDocument:
    document_id: str
    source_name: str
    title: str
    kind: str
    text: str
    revision: str = ""
    valid_from: str = ""
    valid_to: str = ""


@dataclass(frozen=True)
class CorpusFragment:
    fragment_id: str
    document_id: str
    document_title: str
    section: str
    text: str
    start: int
    end: int
    content_kind: str
    revision: str = ""
    valid_from: str = ""
    valid_to: str = ""


@dataclass(frozen=True)
class Corpus:
    documents: tuple[CorpusDocument, ...]
    fragments: tuple[CorpusFragment, ...]
    _by_id: Mapping[str, CorpusDocument] = field(repr=False, compare=False)

    def get_document(self, document_id: str) -> CorpusDocument | None:
        """Return the exact archived revision, or None for an unknown ID."""
        return self._by_id.get(document_id)


def _source_metadata(name: str, text: str) -> tuple[str, str, str]:
    stem = PurePosixPath(name).stem
    label = stem.split("__", 1)[-1]
    kind = "law" if stem.startswith("законы__") else "company"
    if kind == "law":
        match = re.match(r"(.+?)_ст(\d+(?:\.\d+)?)_", label, re.I)
        if match:
            source = match.group(1).replace("_", " ")
            source = {"152ФЗ": "ФЗ № 152-ФЗ", "МРОТ": "ФЗ о МРОТ"}.get(source, f"{source} РФ")
            title = f"{source}, ст. {match.group(2)}"
        else:
            title = label.replace("_", " ")
        # The file contains a current extract plus historical and future notes.
        # Its download date is provenance, not a legal effective date.
        download = re.search(r"(?m)^Дата выгрузки:\s*(\d{2}\.\d{2}\.\d{4})\s*$", text)
        revision = f"выгрузка {download.group(1)}" if download else ""
    else:
        title = label
        revision_match = _REVISION.search(label)
        revision = revision_match.group(1) if revision_match else ""
    return title, kind, revision


def _content_kind(line: str, kind: str) -> str:
    if _INSTRUCTION.search(line):
        return "untrusted_instruction"
    if kind == "law" and _EDITORIAL.match(line):
        return "editorial_note"
    return "source_text"


def _line_spans(text: str):
    cursor = 0
    for line in text.splitlines(keepends=True):
        end = cursor + len(line)
        yield cursor, end, line
        cursor = end


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _split_span(text: str, start: int, end: int, limit: int):
    """Split an oversized source span without altering its characters."""
    while end - start > limit:
        cut = start + limit
        # Prefer a word boundary in the last 40% of the window.
        for pos in range(cut, start + int(limit * 0.6), -1):
            if text[pos - 1].isspace():
                cut = pos
                break
        a, b = _trim_span(text, start, cut)
        if a < b:
            yield a, b
        start = cut
    a, b = _trim_span(text, start, end)
    if a < b:
        yield a, b


def _make_fragments(document: CorpusDocument, limit: int) -> list[CorpusFragment]:
    text = document.text
    result: list[CorpusFragment] = []
    section = ""
    pending_start: int | None = None
    pending_end = 0
    pending_section = ""
    pending_kind = ""

    def emit(start: int, end: int, label: str, content_kind: str) -> None:
        for a, b in _split_span(text, start, end, limit):
            digest = hashlib.sha256(f"{document.document_id}:{a}:{b}".encode("ascii")).hexdigest()[:24]
            result.append(CorpusFragment(
                fragment_id=f"frag-{digest}", document_id=document.document_id,
                document_title=document.title, section=label, text=text[a:b],
                start=a, end=b, content_kind=content_kind,
                revision=document.revision, valid_from=document.valid_from,
                valid_to=document.valid_to,
            ))

    def flush() -> None:
        nonlocal pending_start
        if pending_start is not None:
            emit(pending_start, pending_end, pending_section, pending_kind)
            pending_start = None

    for start, end, line in _line_spans(text):
        if not line.strip():
            flush()
            continue
        match = _SECTION.match(line)
        new_section = match.group("article") or match.group("number") if match else None
        if new_section and new_section != section:
            flush()
            section = new_section
        content_kind = _content_kind(line, document.kind)
        if (pending_start is not None and
                (pending_kind != content_kind or pending_section != section or end - pending_start > limit)):
            flush()
        if pending_start is None:
            pending_start = start
            pending_section = section
            pending_kind = content_kind
        pending_end = end
    flush()
    return result


def load_corpus(archive_path: Path, *, chunk_chars: int = DEFAULT_CHUNK_CHARS) -> Corpus:
    """Read UTF-8 TXT members once into an immutable, in-memory corpus.

    IDs depend on member path and exact bytes, so ZIP order and timestamps do
    not affect them. Malformed archives raise an error instead of hiding data.
    """
    if chunk_chars < 128:
        raise ValueError("chunk_chars must be at least 128")
    documents: list[CorpusDocument] = []
    fragments: list[CorpusFragment] = []
    seen_names: set[str] = set()
    total_bytes = 0
    with zipfile.ZipFile(archive_path) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ValueError("Too many ZIP members")
        for info in sorted(members, key=lambda item: item.filename):
            name = info.filename
            path = PurePosixPath(name)
            if (not name or name.startswith("/") or "\\" in name or
                    any(part in {"", ".", ".."} for part in name.split("/")) or
                    path.suffix.lower() != ".txt"):
                raise ValueError(f"Unsupported ZIP member: {name!r}")
            if name in seen_names:
                raise ValueError(f"Duplicate ZIP member: {name!r}")
            seen_names.add(name)
            total_bytes += info.file_size
            if info.file_size > MAX_DOCUMENT_BYTES or total_bytes > MAX_CORPUS_BYTES:
                raise ValueError("ZIP text exceeds memory limit")
            raw = archive.read(info)
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise ValueError(f"ZIP member is not UTF-8: {name!r}") from exc
            if not text.strip():
                raise ValueError(f"ZIP member is empty: {name!r}")
            digest = hashlib.sha256(name.encode("utf-8") + b"\0" + raw).hexdigest()[:24]
            title, kind, revision = _source_metadata(name, text)
            document = CorpusDocument(f"doc-{digest}", name, title, kind, text, revision)
            documents.append(document)
            fragments.extend(_make_fragments(document, chunk_chars))
    if not documents:
        raise ValueError("ZIP contains no text documents")
    if len({doc.document_id for doc in documents}) != len(documents):
        raise ValueError("Document ID collision")
    if len({fragment.fragment_id for fragment in fragments}) != len(fragments):
        raise ValueError("Fragment ID collision")
    return Corpus(tuple(documents), tuple(fragments),
                  MappingProxyType({doc.document_id: doc for doc in documents}))
