"""Async backend integration checks (filename follows the requested spelling)."""

from __future__ import annotations

import asyncio
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

import api
import solution
from config import Settings
from dependencies import (BackendUnavailable, SearchQuery, UpstreamError,
                          initialize_search)
from search_adapter import create_search_components
from test_search import make_zip, real_archive
from verification import insufficient_result


class CitationLLM:
    """Synthetic response to test plumbing, not legal answer quality."""

    def __init__(self):
        self.calls = 0
        self.first_evidence = None

    async def complete(self, messages):
        self.calls += 1
        if self.calls == 1:
            payload = json.loads(messages[1]["content"])
            self.first_evidence = payload["evidence"][0]
            fragment = self.first_evidence
            return json.dumps({
                "status": "answer",
                "text": "Synthetic integration response",
                "claims": [{
                    "text": "Synthetic claim citing a retrieved fragment",
                    "fragment_ids": [fragment["fragment_id"]],
                    "quotes": {fragment["fragment_id"]: fragment["text"][:40]},
                }],
            }, ensure_ascii=False)
        return '{"supported": true}'


class NeutralLLM:
    text = "Недостаточно сведений"

    async def complete(self, messages):
        return json.dumps({"status": "insufficient", "text": self.text},
                          ensure_ascii=False)


class AdapterUnitTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_and_invalid_archive_are_distinct_errors(self):
        with self.assertRaises(BackendUnavailable):
            await create_search_components(Path("missing-knowledge-base.zip"), Settings())
        with self.assertRaises(UpstreamError):
            await create_search_components(io.BytesIO(b"not a ZIP"), Settings())
        with self.assertRaises(UpstreamError):
            await create_search_components(make_zip([("bad.txt", b"\xff")]), Settings())

    async def test_one_index_for_multiple_requests_and_document_lookup(self):
        archive = make_zip([("source.txt", b"1. First provision\n2. Second provision")])
        # Patch the actual adapter import; the archive must be read only once.
        from search_service import load_search_service
        with patch("search_adapter.load_search_service", wraps=load_search_service) as loader:
            components = await create_search_components(archive, Settings(top_k=1))
            first = await components.evidence(SearchQuery("First"))
            second = await components.evidence(SearchQuery("Second"))
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        document = await components.documents(first[0].document_id)
        self.assertIn(first[0].text, document.text)
        self.assertIsNone(first[0].start)
        self.assertIsNone(first[0].end)
        self.assertIsNone(await components.documents("unknown"))
        self.assertEqual(await components.evidence(SearchQuery("zzznothing")), [])
        with self.assertRaises(UpstreamError):
            await components.evidence(SearchQuery("First", filters=(("valid_on", "2024-01-01"),)))


class RealIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.archive = real_archive()
        if self.archive is None:
            self.skipTest("Set PSB_TEST_ARCHIVE to run checks on the real knowledge base")

    async def test_evidence_document_and_parallel_calls(self):
        components = await initialize_search(self.archive, Settings(top_k=5))
        query = SearchQuery("Какой срок оплаты по договору П-17/2024 с Альфа-Трейд?")
        hits = await components.evidence(query)
        self.assertEqual(hits[0].section, "4.2")
        self.assertLessEqual(len(hits), 5)
        self.assertEqual(len({item.fragment_id for item in hits}), len(hits))
        for item in hits:
            document = await components.documents(item.document_id)
            self.assertIn(item.text, document.text)
            self.assertIsNone(item.start)
            self.assertIsNone(item.end)
        parallel = await asyncio.gather(*(components.evidence(query) for _ in range(8)))
        self.assertTrue(all([item.fragment_id for item in batch] ==
                            [item.fragment_id for item in hits] for batch in parallel))

    async def test_regulation_expansion_and_followup_at_backend_boundary(self):
        settings = Settings(top_k=10, max_context_chars=4_000)
        components = await initialize_search(self.archive, settings)
        for year, revision in ((2024, "2023"), (2026, "2025")):
            with self.subTest(year=year):
                hits = await components.evidence(SearchQuery(
                    f"Какой порог согласования договора действовал в {year} году?"))
                regulation = [item for item in hits if
                              "Регламент согласования договоров" in item.document_title]
                self.assertTrue({"2.1", "2.2", "2.3", "2.4"}.issubset(
                    {item.section for item in regulation}))
                self.assertEqual({item.revision for item in regulation}, {revision})
                self.assertLessEqual(sum(len(json.dumps(item.model_dump(), ensure_ascii=False))
                                         for item in hits), settings.max_context_chars)
                self.assertEqual(len({item.fragment_id for item in hits}), len(hits))
                for item in regulation:
                    source = await components.documents(item.document_id)
                    self.assertIn(item.text, source.text)
        followup = await components.evidence(SearchQuery(
            "А с какого момента начинает течь этот срок?",
            user_context=("Какой срок оплаты по договору П-17/2024 с Альфа-Трейд?",)))
        self.assertEqual(followup[0].section, "4.2")

    async def test_api_startup_once_and_source_route(self):
        llm = CitationLLM()
        with patch("api.initialize_search", wraps=initialize_search) as loader:
            app = api.create_app(archive_path=self.archive, llm=llm)
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                             base_url="http://test") as client:
                    response = await client.post("/api/chat", json={
                        "question": "А какой срок оплаты?",
                        "history": [
                            {"role": "user", "content": "Договор П-17/2024 с Альфа-Трейд"},
                            {"role": "assistant", "content": "Предыдущий ответ не является источником"},
                        ],
                    })
                    self.assertEqual(response.status_code, 200)
                    payload = response.json()
                    self.assertEqual(payload["status"], "answer")
                    self.assertEqual(llm.first_evidence["section"], "4.2")
                    source = payload["fragments"][0]
                    self.assertEqual(payload["claims"][0]["fragment_ids"], [source["fragment_id"]])
                    document = await client.get("/api/documents/" + source["document_id"])
                    self.assertEqual(document.status_code, 200)
                    self.assertIn(source["text"], document.json()["text"])
                    missing = await client.get("/api/documents/unknown")
                    self.assertEqual(missing.status_code, 404)
            self.assertEqual(loader.call_count, 1)
        self.assertEqual(llm.calls, 2)

    async def test_solution_default_factory_and_independent_ids(self):
        captured = []
        questions = [
            ("a", "Какой срок оплаты по договору П-17/2024?"),
            ("b", "несуществующийквантовыйтермин"),
        ]
        with (patch("solution.input_paths", return_value=(Path("virtual-questions.csv"), self.archive)),
              patch("solution.read_questions", return_value=questions),
              patch("solution.write_submission", side_effect=lambda path, rows: captured.extend(rows)),
              patch("solution.initialize_search", wraps=initialize_search) as loader):
            await solution.run(data_dir=Path("."), output_path=Path("virtual-submission.csv"),
                               settings=Settings(), llm=NeutralLLM())
        self.assertEqual(loader.call_count, 1)
        self.assertEqual([case_id for case_id, _ in captured], ["a", "b"])
        expected = solution.format_submission_answer(insufficient_result())
        self.assertTrue(all(answer == expected for _, answer in captured))
        self.assertTrue(all(NeutralLLM.text not in answer for _, answer in captured))


if __name__ == "__main__":
    unittest.main()
