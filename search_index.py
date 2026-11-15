"""Read-only lexical index for exact, source-grounded fragment retrieval.

No external model, network request, or persisted index is required. Build the
index once per corpus, then share it between concurrent read-only searches.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from search_corpus import Corpus, CorpusFragment


_TOKEN = re.compile(r"[^\W_]+(?:[./-][^\W_]+)*", re.UNICODE)
_STOP = frozenset({
    "а", "без", "бы", "в", "во", "все", "вы", "где", "да", "для", "до", "его", "ее",
    "если", "же", "за", "и", "из", "или", "им", "их", "к", "как", "какая", "какие",
    "какой", "кем", "когда", "ли", "мы", "на", "над", "нам", "не", "него", "нее",
    "но", "о", "об", "от", "по", "под", "при", "про", "с", "со", "так", "то", "у",
    "что", "чем", "это", "этот", "я", "рф", "ст", "статья", "пункт", "документ",
})
_SUFFIXES = tuple(sorted({
    "иями", "ями", "ами", "ения", "ению", "ением", "ений", "ение", "остей",
    "ость", "остью", "ового", "овому", "овыми", "овый", "овая", "овые", "овых",
    "ального", "альная", "альные", "альных", "ация", "ации", "ацию", "ацией",
    "овать", "ованный", "ование", "ий", "ый", "ой", "ая", "яя", "ые", "ие",
    "ого", "его", "ому", "ему", "ами", "ями", "ов", "ев", "ам", "ям", "ах",
    "ях", "ом", "ем", "ей", "ия", "ью", "ию", "ие", "ии", "иям", "ий", "ий",
    "ы", "и", "а", "я", "у", "ю", "е", "о",
}, key=len, reverse=True))
_RELATED_WORDS = (
    ("оплата", "оплаты", "оплачивает", "оплачиваются", "платеж", "расчеты"),
    ("доставка", "доставляется", "поставка", "отгрузка"),
    ("согласование", "согласуется", "согласовывается"),
)


def normalize_token(token: str) -> str:
    """Normalize case, ё and separator variants; stem only plain Cyrillic words."""
    token = token.casefold().replace("ё", "е").replace("/", "-").replace(".", "-")
    if "-" not in token and len(token) >= 6 and token.isalpha():
        for suffix in _SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= 4:
                return token[:-len(suffix)]
    return token


_NORMALIZED_STOP = frozenset(normalize_token(word) for word in _STOP)
_RELATED_TERMS = tuple(frozenset(normalize_token(word) for word in group)
                       for group in _RELATED_WORDS)
_TIMING_TERMS = frozenset(normalize_token(word) for word in
                          ("срок", "течение", "позднее"))


def terms(text: str) -> tuple[str, ...]:
    """Return lexical terms; contract numbers remain single searchable tokens."""
    return tuple(
        normalized for match in _TOKEN.finditer(text)
        if (normalized := normalize_token(match.group())) not in _NORMALIZED_STOP and len(normalized) > 1
    )


@dataclass(frozen=True)
class _Entry:
    fragment: CorpusFragment
    length: int
    title_terms: frozenset[str]
    content_terms: frozenset[str]


@dataclass(frozen=True)
class SearchIndex:
    """BM25 postings and exact-title terms, all immutable after construction."""

    entries: tuple[_Entry, ...]
    postings: Mapping[str, tuple[tuple[int, int], ...]]
    idf: Mapping[str, float]
    average_length: float

    @classmethod
    def build(cls, corpus: Corpus) -> SearchIndex:
        entries: list[_Entry] = []
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for fragment in corpus.fragments:
            if fragment.content_kind == "untrusted_instruction":
                continue
            counts = Counter(terms(fragment.text))
            title_terms = frozenset(terms(fragment.document_title))
            if not counts and not title_terms:
                continue
            index = len(entries)
            entries.append(_Entry(fragment, sum(counts.values()), title_terms, frozenset(counts)))
            for term, count in counts.items():
                postings[term].append((index, count))
        count = len(entries)
        average = sum(entry.length for entry in entries) / count if count else 1.0
        idf = {term: math.log1p((count - len(rows) + 0.5) / (len(rows) + 0.5))
               for term, rows in postings.items()}
        return cls(
            tuple(entries),
            MappingProxyType({term: tuple(rows) for term, rows in postings.items()}),
            MappingProxyType(idf), average,
        )

    def score(self, query: str, context: tuple[str, ...] = ()) -> dict[int, float]:
        """Score current question first; prior user messages provide weak hints."""
        query_terms = tuple(dict.fromkeys(terms(query)))
        query_codes = {term for term in query_terms if "-" in term and
                       any(ch.isdigit() for ch in term) and any(ch.isalpha() for ch in term)}
        prior_terms = tuple(dict.fromkeys(terms(context[-1]))) if context else ()
        prior_codes = {term for term in prior_terms if "-" in term and
                       any(ch.isdigit() for ch in term) and any(ch.isalpha() for ch in term)}
        # A new explicit contract wins. Multiple contracts in the previous user
        # message are ambiguous, so neither its identity nor its topic is carried.
        use_context = bool(context) and len(prior_codes) == 1 and not (
            query_codes and query_codes != prior_codes)
        context_terms = prior_terms[:80] if use_context else ()
        weights = {term: 0.25 for term in context_terms}
        related: set[str] = set()
        current_topics = {position for position, group in enumerate(_RELATED_TERMS)
                          if any(term in group for term in query_terms)}
        carried_topics = set()
        if use_context and not current_topics:
            prior_topics = {position for position, group in enumerate(_RELATED_TERMS)
                            if any(term in group for term in prior_terms)}
            if len(prior_topics) == 1:
                carried_topics = prior_topics
        for position in current_topics | carried_topics:
            related.update(_RELATED_TERMS[position])
        weights.update({term: 0.4 for term in related})
        weights.update({term: 1.0 for term in query_terms})
        scores: dict[int, float] = defaultdict(float)
        for term, weight in weights.items():
            for index, frequency in self.postings.get(term, ()):
                entry = self.entries[index]
                denom = frequency + 1.2 * (0.25 + 0.75 * entry.length / self.average_length)
                scores[index] += weight * self.idf[term] * frequency * 2.2 / denom
        # Titles and exact identifiers are important in this corpus: several
        # contracts have near-identical counterparties but different numbers.
        identifiers = [term for term in query_terms if any(ch.isdigit() for ch in term)]
        contract_codes = query_codes.copy()
        anchored_by_context = not contract_codes and use_context
        if anchored_by_context:
            contract_codes = prior_codes
        law_codes = set(query_terms) & {"нк", "тк", "коап"}
        article_numbers = {term for term in identifiers if term.isdigit() and len(term) <= 3}
        asks_payment_deadline = (
            (normalize_token("срок") in query_terms or
             any(word in query.casefold().replace("ё", "е") for word in
                 ("отсчет", "начинает", "начинается", "течет"))) and
            0 in (current_topics | carried_topics)
        )
        for index, entry in enumerate(self.entries):
            title_matches = sum(weights[term] for term in entry.title_terms if term in query_terms)
            if title_matches:
                scores[index] += 0.6 * title_matches
            exact_ids = sum(1 for term in identifiers if term in entry.title_terms)
            if exact_ids:
                scores[index] += 3.0 * exact_ids
            # Evidence about the requested subject matters more than a title
            # alone. Related terms help with simple inflections such as
            # "оплата" vs "оплачивает" without rewriting source text.
            topic_terms = set(query_terms) - entry.title_terms - set(identifiers)
            direct_topic = len(topic_terms & entry.content_terms)
            related_topic = len((related - set(query_terms)) & entry.content_terms)
            scores[index] += 2.0 * direct_topic + 0.8 * related_topic
            if (asks_payment_deadline and
                    entry.content_terms & _RELATED_TERMS[0] and
                    entry.content_terms & _TIMING_TERMS):
                scores[index] += 12.0
            if contract_codes:
                title_codes = {term for term in entry.title_terms
                               if "-" in term and any(ch.isdigit() for ch in term)
                               and any(ch.isalpha() for ch in term)}
                if title_codes & contract_codes:
                    scores[index] *= 1.3 if anchored_by_context else 1.4
                elif title_codes:
                    scores[index] *= 0.2
                else:
                    scores[index] *= 0.5 if anchored_by_context else 0.6
            if law_codes and article_numbers:
                title_law_codes = entry.title_terms & {"нк", "тк", "коап"}
                title_articles = {term for term in entry.title_terms
                                  if term.isdigit() and len(term) <= 3}
                if title_law_codes & law_codes and title_articles & article_numbers:
                    scores[index] *= 1.5
                elif title_law_codes:
                    scores[index] *= 0.35
                else:
                    scores[index] *= 0.55
            if entry.fragment.content_kind == "editorial_note" and not any(
                word in query.casefold() for word in ("редакц", "изменен", "изменени", "истори", "вступ", "утрат")
            ):
                scores[index] *= 0.25
        return {index: score for index, score in scores.items() if score > 0}
