"""Synthetic fixtures only; no real knowledge-base content or proxy requests."""
import json
import unittest
from unittest.mock import patch
import httpx
from pydantic import ValidationError
from assistant import Assistant
from config import Settings
from dependencies import BackendUnavailable, UpstreamError, UpstreamTimeout, SearchQuery, prepare_search_query
from llm_client import OpenAICompatibleClient
from schemas import AssistantResult, ChatRequest, Draft, Fragment


SOURCE = Fragment(fragment_id="test-f1", document_id="test-doc", document_title="ТЕСТОВЫЙ документ",
                  section="п. 1", text="ТЕСТ: требуется согласование.")


def draft(source_id="test-f1", quote="требуется согласование."):
    return {"status": "answer", "text": "Не использовать этот непроверенный текст",
            "claims": [{"text": "ТЕСТ: нужно согласование (ТЕСТОВЫЙ документ, п. 1).",
                        "fragment_ids": [source_id], "quotes": {source_id: quote}}]}


class FakeEvidence:
    def __init__(self, fragments=None):
        self.fragments = [SOURCE] if fragments is None else fragments
        self.requests = []

    async def __call__(self, request):
        self.requests.append(request)
        return self.fragments


class FakeLLM:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.messages = []

    async def complete(self, messages):
        self.messages.append(messages)
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value


class SchemaTests(unittest.TestCase):
    def test_invalid_requests(self):
        for data in ({"question": " "}, {"question": "x", "history": [{"role": "system", "content": "x"}]},
                     {"question": "x", "unexpected": True}):
            with self.subTest(data=data), self.assertRaises(ValidationError):
                ChatRequest.model_validate(data)

    def test_sources_and_offsets(self):
        with self.assertRaises(ValidationError):
            Fragment(**{**SOURCE.model_dump(), "start": 0})
        with self.assertRaises(ValidationError):
            AssistantResult(status="answer", text="x", claims=[{"text": "x", "fragment_ids": ["fake"]}])
        for timeout in (0, 60, float("nan")):
            with self.assertRaises(ValueError):
                Settings(request_timeout=timeout)


class AssistantTests(unittest.IsolatedAsyncioTestCase):
    async def test_history_and_verified_answer(self):
        evidence, llm = FakeEvidence(), FakeLLM(draft(), {"supported": True})
        request = ChatRequest(question="Текущий вопрос", history=[{"role": "user", "content": "Предыдущий вопрос"},
                                                               {"role": "assistant", "content": "Старая гипотеза"}])
        result = await Assistant(evidence, llm, Settings()).answer(request)
        self.assertEqual(result.status, "answer")
        self.assertEqual(evidence.requests[0], prepare_search_query(request))
        payload = json.loads(llm.messages[0][1]["content"])
        self.assertEqual(payload["history"], [m.model_dump() for m in request.history])
        self.assertEqual(llm.messages[0][1]["content"].count("Текущий вопрос"), 1)
        self.assertNotIn("непроверенный", result.text)
        self.assertEqual(result.fragments, [SOURCE])
        self.assertEqual(result.claims[0].fragment_ids, [SOURCE.fragment_id])

    async def test_non_answers(self):
        for status in ("clarification", "insufficient"):
            with self.subTest(status=status):
                llm = FakeLLM({"status": status, "text": "Укажите дату"})
                result = await Assistant(FakeEvidence([]), llm, Settings()).answer(ChatRequest(question="Вопрос"))
                self.assertEqual(result.status, status)
                self.assertEqual(result.fragments, [])

    async def test_invalid_citations_and_missing_sources(self):
        for fragments, value in (([SOURCE], draft("invented")), ([SOURCE], draft(quote="Выдуманная цитата")),
                                 ([], draft())):
            llm = FakeLLM(value)
            result = await Assistant(FakeEvidence(fragments), llm, Settings()).answer(ChatRequest(question="x"))
            self.assertEqual(result.status, "insufficient")
            self.assertEqual(result.claims, [])
            self.assertEqual(len(llm.messages), 1)

    async def test_semantic_rejection(self):
        llm = FakeLLM(draft(), {"supported": False})
        result = await Assistant(FakeEvidence(), llm, Settings()).answer(ChatRequest(question="x"))
        self.assertEqual(result.status, "insufficient")
        self.assertEqual(len(llm.messages), 2)
        self.assertEqual(result.claims, [])
        self.assertEqual(result.fragments, [])

    async def test_malformed_output(self):
        with self.assertRaises(UpstreamError):
            await Assistant(FakeEvidence(), FakeLLM("not JSON"), Settings()).answer(ChatRequest(question="x"))

    async def test_duplicate_sources(self):
        with self.assertRaises(UpstreamError):
            await Assistant(FakeEvidence([SOURCE, SOURCE]), FakeLLM(), Settings()).answer(ChatRequest(question="x"))

    async def test_context_budget(self):
        llm = FakeLLM({"status": "insufficient", "text": "Нет оснований"})
        await Assistant(FakeEvidence(), llm, Settings(max_context_chars=1)).answer(ChatRequest(question="x"))
        self.assertEqual(json.loads(llm.messages[0][1]["content"])["evidence"], [])


class PromptPolicyTests(unittest.IsolatedAsyncioTestCase):
    """Synthetic LLM outputs verify the prompt contract and answer workflow."""

    async def test_confirmed_general_order_with_unknown_amount(self):
        general = Fragment(
            fragment_id="general", document_id="reg-2023",
            document_title="ТЕСТОВЫЙ регламент, редакция 2023", section="2.1",
            text="2.1. Проект договора согласует юридический отдел.", revision="2023")
        threshold = Fragment(
            fragment_id="threshold", document_id="reg-2023",
            document_title="ТЕСТОВЫЙ регламент, редакция 2023", section="2.3",
            text="2.3. При сумме свыше 100 рублей требуется согласование комиссии.",
            revision="2023")
        llm = FakeLLM({
            "status": "answer", "text": "Неиспользуемый текст",
            "claims": [
                {"text": "По тестовому регламенту проект договора согласует юридический отдел (п. 2.1).",
                 "fragment_ids": ["general"],
                 "quotes": {"general": "Проект договора согласует юридический отдел."}},
                {"text": "При сумме свыше 100 рублей требуется согласование комиссии (п. 2.3).",
                 "fragment_ids": ["threshold"],
                 "quotes": {"threshold": "При сумме свыше 100 рублей требуется согласование комиссии."}},
            ],
        }, {"supported": True})
        result = await Assistant(FakeEvidence([general, threshold]), llm, Settings()).answer(
            ChatRequest(question="Каков общий порядок согласования договора в 2024 году?"))
        self.assertEqual(result.status, "answer")
        self.assertEqual([claim.fragment_ids for claim in result.claims],
                         [["general"], ["threshold"]])
        self.assertEqual([item.fragment_id for item in result.fragments],
                         ["general", "threshold"])
        system = llm.messages[0][0]["content"]
        self.assertIn("сначала изложи подтверждённую часть", system)
        self.assertIn("Пороговые требования и", system)
        self.assertIn("не утверждая, что они выполнены", system)
        self.assertIn("подтверждай собственной", system)

    async def test_missing_essential_event_date_requires_clarification(self):
        timing = Fragment(
            fragment_id="timing", document_id="test-contract",
            document_title="ТЕСТОВЫЙ договор", section="4.2",
            text="4.2. Оплата в течение 30 дней после получения товара.")
        llm = FakeLLM({"status": "clarification",
                       "text": "Когда был получен товар?", "claims": [],
                       "clarification_category": "event_date"})
        result = await Assistant(FakeEvidence([timing]), llm, Settings()).answer(
            ChatRequest(question="Какого числа истекает срок оплаты?"))
        self.assertEqual(result.status, "clarification")
        self.assertEqual(result.claims, [])
        self.assertEqual(len(llm.messages), 1)
        self.assertEqual(result.text, "Укажите дату события, о котором идёт речь.")
        self.assertIn("нельзя дать даже частичный", llm.messages[0][0]["content"])

    async def test_unresolved_conflicting_documents_remain_insufficient(self):
        first = Fragment(fragment_id="first", document_id="one",
                         document_title="ТЕСТОВОЕ положение 1", section="1",
                         text="Для этой операции требуется согласование.")
        second = Fragment(fragment_id="second", document_id="two",
                          document_title="ТЕСТОВОЕ положение 2", section="1",
                          text="Для этой операции согласование не требуется.")
        llm = FakeLLM({"status": "insufficient",
                       "text": "Документы противоречат друг другу.", "claims": []})
        result = await Assistant(FakeEvidence([first, second]), llm, Settings()).answer(
            ChatRequest(question="Требуется ли согласование для этой операции?"))
        self.assertEqual(result.status, "insufficient")
        self.assertEqual(result.fragments, [])
        self.assertIn("При неразрешимом противоречии", llm.messages[0][0]["content"])

    async def test_specific_approval_requires_unknown_amount(self):
        threshold = Fragment(
            fragment_id="threshold", document_id="reg-2023",
            document_title="ТЕСТОВЫЙ регламент, редакция 2023", section="2.3",
            text="2.3. При сумме свыше 100 рублей требуется согласование комиссии.",
            revision="2023")
        llm = FakeLLM({"status": "clarification",
                       "text": "Какова сумма вашего договора?", "claims": [],
                       "clarification_category": "amount"})
        result = await Assistant(FakeEvidence([threshold]), llm, Settings()).answer(
            ChatRequest(question="Требуется ли согласование комиссии именно для моего договора?"))
        self.assertEqual(result.status, "clarification")
        self.assertEqual(result.claims, [])
        self.assertEqual(result.text, "Какова сумма сделки и в какой валюте она указана?")
        self.assertIn("Не придумывай сумму", llm.messages[0][0]["content"])
        self.assertIn("общий порядок не заменяет такое решение",
                      llm.messages[0][0]["content"])


class SelfContainedClaimTests(unittest.IsolatedAsyncioTestCase):
    """Invented source fixtures and canned verdicts, NOT real DeepSeek/legal validation."""

    def source(self, source_id, text, document_id="test-reg"):
        return Fragment(fragment_id=source_id, document_id=document_id,
                        document_title="ТЕСТОВЫЙ документ " + document_id, text=text)

    def claim(self, text, *sources):
        return {"text": text, "fragment_ids": [s.fragment_id for s in sources],
                "quotes": {s.fragment_id: s.text for s in sources}}

    async def run_claims(self, question, sources, claims, *, supported=True):
        llm = FakeLLM({"status": "answer", "text": "", "claims": claims},
                      {"supported": supported})
        result = await Assistant(FakeEvidence(sources), llm, Settings()).answer(ChatRequest(question=question))
        system = llm.messages[0][0]["content"]
        for instruction in ("Каждый claim должен быть самодостаточным", "необходимые реальные fragment_ids",
                            "Не расширяй область применения частного требования", "не приписывай условия письма"):
            self.assertIn(instruction, system)
        return result, llm

    async def test_d_general_order_retains_conditional_thresholds(self):
        sources = [self.source(str(i), text) for i, text in enumerate((
            "2.1. Проект согласуют юридический отдел и главный бухгалтер.",
            "2.2. Юридический отдел согласует за 3 рабочих дня, главный бухгалтер — за 2 рабочих дня.",
            "2.3. При сумме свыше 1 млн рублей требуется дополнительное согласование финансового директора.",
            "2.4. При сумме свыше 10 млн рублей требуется одобрение генерального директора до подписания.",
        ))]
        claims = [self.claim("В ТЕСТОВОЙ редакции предусмотрен следующий общий порядок; "
                             "её применимость к событию 2024 года не установлена: " + s.text, s)
                  for s in sources]
        result, llm = await self.run_claims("ТЕСТ: порядок согласования для события 2024 года?", sources, claims)
        self.assertEqual(result.status, "answer")
        self.assertEqual(result.text, "\n\n".join(c["text"] for c in claims))
        self.assertEqual(len(llm.messages), 2)
        checked = json.loads(llm.messages[1][1]["content"])
        self.assertEqual(checked["draft"]["claims"], claims)
        self.assertEqual(result.claims[2].fragment_ids, [sources[2].fragment_id])

    async def test_d_cancellation_alone_cannot_establish_applicability(self):
        source = self.source("end", "ТЕСТ: редакция утратила силу 01.03.2025.")
        claims = [self.claim("ТЕСТ: эта редакция безусловно действовала в 2024 году.", source)]
        # Exact citation passes; semantic rejection remains authoritative.
        from verification import validate_citations
        self.assertTrue(validate_citations(Draft(status="answer", text="", claims=claims), [source]))
        result, llm = await self.run_claims("ТЕСТ: редакция для 2024 года?", [source], claims, supported=False)
        self.assertIn("только по дате её отмены", llm.messages[0][0]["content"])
        self.assertIn("дата утверждения не равнозначна дате вступления в силу", llm.messages[0][0]["content"])
        self.assertEqual(result.status, "insufficient")
        self.assertEqual(result.claims, [])
        self.assertEqual(len(llm.messages), 2)

    async def test_d_date_conclusion_can_cite_multiple_fragments(self):
        # Explicit effective date exists ONLY in this synthetic fixture.
        start = self.source("start", "ТЕСТ: редакция вступает в силу 01.02.2023.")
        end = self.source("end", "ТЕСТ: редакция утратила силу 01.03.2025.")
        claims = [self.claim("ТЕСТ: указан период действия с 01.02.2023 до утраты силы 01.03.2025.", start, end)]
        result, llm = await self.run_claims("ТЕСТ: какой период указан?", [start, end], claims)
        self.assertEqual(result.status, "answer")
        self.assertEqual(result.claims[0].fragment_ids, ["start", "end"])
        self.assertEqual(result.fragments, [start, end])
        self.assertEqual(len(llm.messages), 2)

    def sanctions_fixture(self):
        scope = self.source("scope", "ТЕСТОВОЕ письмо К-12: с 21.04.2025 иностранные контрагенты "
                            "проходят санкционную проверку до подписания договора.", "test-letter")
        timing = self.source("timing", "ТЕСТ: срок проверки — 3 рабочих дня с получения запроса; "
                             "заключение размещается в СЭД.", "test-letter")
        appendix = self.source("appendix", "ТЕСТОВОЕ приложение: проверяется страна регистрации.", "test-letter")
        regulation = self.source("reg54", "ТЕСТОВАЯ редакция 2025, п. 5.4: для иностранного контрагента "
                                 "проводятся юридическая проверка и валютный контроль.")
        return scope, timing, appendix, regulation

    async def test_f_scope_repeated_and_letter_separate_from_regulation(self):
        scope, timing, appendix, regulation = self.sanctions_fixture()
        claims = [
            self.claim("По ТЕСТОВОМУ письму К-12 с 21.04.2025 иностранный контрагент проходит "
                       "санкционную проверку до подписания договора.", scope),
            self.claim("По ТЕСТОВОМУ письму К-12 с 21.04.2025 проверка иностранного контрагента "
                       "занимает 3 рабочих дня с получения запроса; заключение размещается в СЭД.", scope, timing),
            self.claim("По приложению к ТЕСТОВОМУ письму К-12 с 21.04.2025 при санкционной проверке "
                       "иностранного контрагента проверяется страна регистрации.", scope, appendix),
            self.claim("По п. 5.4 ТЕСТОВОЙ редакции 2025 для иностранного контрагента проводятся "
                       "юридическая проверка и валютный контроль.", regulation),
        ]
        result, llm = await self.run_claims("ТЕСТ: требования санкционной проверки?",
                                           [scope, timing, appendix, regulation], claims)
        self.assertEqual(result.status, "answer")
        self.assertEqual([c.fragment_ids for c in result.claims],
                         [["scope"], ["scope", "timing"], ["scope", "appendix"], ["reg54"]])
        self.assertTrue(all("иностранн" in c.text for c in result.claims))
        self.assertEqual(len(llm.messages), 2)

    async def test_f_generalization_or_mixed_documents_blocks_entire_answer(self):
        scope, timing, _, regulation = self.sanctions_fixture()
        good = self.claim("По ТЕСТОВОМУ письму с 21.04.2025 иностранный контрагент проходит "
                          "санкционную проверку до подписания договора.", scope)
        for bad in (self.claim("Проверка всех контрагентов занимает 3 рабочих дня.", scope, timing),
                    self.claim("Пункт 5.4 устанавливает срок санкционной проверки 3 рабочих дня.", regulation)):
            with self.subTest(bad=bad):
                result, llm = await self.run_claims("ТЕСТ: проверка?", [scope, timing, regulation],
                                                   [good, bad], supported=False)
                self.assertEqual(result.status, "insufficient")
                self.assertEqual(result.claims, [])
                self.assertEqual(result.fragments, [])
                self.assertEqual(len(llm.messages), 2)

    async def test_a_payment_unchanged_requires_contract_and_amendment_quotes(self):
        contract = self.source("payment", "ТЕСТОВЫЙ договор П-17/2024, п. 4.2: оплата в течение "
                               "30 календарных дней после подписания УПД.", "test-contract")
        amendment = self.source("amendment", "ТЕСТОВОЕ допсоглашение № 1: изменяется адрес доставки; "
                                "остальные условия договора сохраняются.", "test-amendment")
        claim = self.claim("По ТЕСТОВОМУ договору П-17/2024 оплата производится в течение 30 календарных "
                           "дней после подписания УПД. ТЕСТОВОЕ допсоглашение № 1 меняет адрес доставки "
                           "и сохраняет остальные условия, включая этот срок оплаты.", contract, amendment)
        result, llm = await self.run_claims("ТЕСТ: срок оплаты?", [contract, amendment], [claim])
        self.assertEqual(result.status, "answer")
        self.assertEqual(result.claims[0].fragment_ids, ["payment", "amendment"])
        self.assertEqual(len(llm.messages), 2)
        # Missing required quote must fail before the semantic call.
        claim["quotes"].pop("amendment")
        result, llm = await self.run_claims("ТЕСТ: срок оплаты?", [contract, amendment], [claim])
        self.assertEqual(result.status, "insufficient")
        self.assertEqual(result.claims, [])
        self.assertEqual(len(llm.messages), 1)


class NonAnswerSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_status_switch_cannot_publish_unverified_text_or_claims(self):
        # Same unverified conclusion must not escape by changing its status.
        unsafe = "ТЕСТ: по выдуманному документу все сделки требуют оплаты за 999 дней."
        for status in ("clarification", "insufficient"):
            for claims in ([], draft("invented", "invented")["claims"]):
                for fragments in ([], [SOURCE]):
                    with self.subTest(status=status, claims=bool(claims), evidence=bool(fragments)):
                        value = {"status": status, "text": unsafe, "claims": claims,
                                 "clarification_category": "amount"}
                        llm = FakeLLM(value)
                        result = await Assistant(FakeEvidence(fragments), llm, Settings()).answer(
                            ChatRequest(question="Тест"))
                        self.assertNotIn(unsafe, result.text)
                        self.assertNotIn("invented", result.model_dump_json())
                        self.assertEqual(result.claims, [])
                        self.assertEqual(result.fragments, [])
                        self.assertEqual(len(llm.messages), 1)
                        if status == "insufficient":
                            from verification import insufficient_result
                            self.assertEqual(result, insufficient_result())
                        else:
                            self.assertEqual(result.text, "Какова сумма сделки и в какой валюте она указана?")

    async def test_controlled_categories_and_unknown_contract(self):
        questions = {
            "contract": "Укажите номер или название договора, о котором идёт речь.",
            "event_date": "Укажите дату события, о котором идёт речь.",
            "amount": "Какова сумма сделки и в какой валюте она указана?",
            "operation_type": "Какой тип операции вы рассматриваете?",
            "counterparty_type": "Уточните тип контрагента, о котором идёт речь.",
            "circumstances": "Опишите обстоятельства события, по которому задан вопрос.",
        }
        for category, expected in questions.items():
            with self.subTest(category=category):
                llm = FakeLLM({"status": "clarification", "text": "Не публиковать",
                               "clarification_category": category})
                result = await Assistant(FakeEvidence([]), llm, Settings()).answer(
                    ChatRequest(question="Какой срок оплаты по неизвестному договору?"))
                self.assertEqual(result.status, "clarification")
                self.assertEqual(result.text, expected)
                self.assertEqual(len(llm.messages), 1)
                self.assertEqual(set(result.model_dump()), {"status", "text", "claims", "fragments"})

    async def test_unknown_malformed_and_legacy_categories_use_safe_fallback(self):
        for value in ({}, {"clarification_category": None},
                      {"clarification_category": "invented rule"},
                      {"clarification_category": ["amount"]},
                      {"clarification_category": {"amount": "invented rule"}},
                      {"clarification_category": 123}):
            with self.subTest(value=value):
                llm = FakeLLM({"status": "clarification", "text": "НЕБЕЗОПАСНЫЙ ТЕКСТ", **value})
                result = await Assistant(FakeEvidence([]), llm, Settings()).answer(ChatRequest(question="Тест"))
                self.assertEqual(result.status, "clarification")
                self.assertEqual(result.text, "Уточните, пожалуйста, обстоятельства и предмет вашего вопроса.")
                self.assertEqual(len(llm.messages), 1)

    async def test_answer_still_requires_citations_then_support_check(self):
        for valid_quote, supported in ((False, True), (True, False), (True, True)):
            with self.subTest(valid_quote=valid_quote, supported=supported):
                value = draft() if valid_quote else draft(quote="Нет такой цитаты")
                value["clarification_category"] = "amount"
                llm = FakeLLM(value, {"supported": supported})
                result = await Assistant(FakeEvidence(), llm, Settings()).answer(ChatRequest(question="Тест"))
                self.assertEqual(len(llm.messages), 2 if valid_quote else 1)
                if valid_quote and supported:
                    self.assertEqual(result.status, "answer")
                    self.assertEqual(result.text, value["claims"][0]["text"])
                    self.assertEqual(result.fragments, [SOURCE])
                    check_payload = json.loads(llm.messages[1][1]["content"])
                    self.assertEqual(check_payload["draft"]["claims"], value["claims"])
                else:
                    self.assertEqual(result.status, "insufficient")
                    self.assertEqual(result.claims, [])
                    self.assertEqual(result.fragments, [])


class EmptyDraftTextTests(unittest.IsolatedAsyncioTestCase):
    async def chat(self, value, *, supported=True):
        from api import create_app
        llm = FakeLLM(value, {"supported": supported})
        app = create_app(evidence=FakeEvidence(), llm=llm)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/chat", json={"question": "ТЕСТОВЫЙ вопрос"})
        return response, llm

    async def test_blank_clarifications_use_backend_questions(self):
        cases = (
            ("", "contract", "Укажите номер или название договора, о котором идёт речь."),
            ("   ", "amount", "Какова сумма сделки и в какой валюте она указана?"),
            ("", "unknown", "Уточните, пожалуйста, обстоятельства и предмет вашего вопроса."),
            ("", "event_date", "Укажите дату события, о котором идёт речь."),
        )
        for text, category, expected in cases:
            with self.subTest(text=text, category=category):
                response, llm = await self.chat({"status": "clarification", "text": text,
                                                 "clarification_category": category})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"status": "clarification", "text": expected,
                                                   "claims": [], "fragments": []})
                self.assertEqual(len(llm.messages), 1)

    async def test_blank_insufficient_uses_backend_message(self):
        from verification import insufficient_result
        for text in ("", "   "):
            with self.subTest(text=text):
                response, llm = await self.chat({"status": "insufficient", "text": text,
                                                 "claims": draft("invented")["claims"]})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), insufficient_result().model_dump())
                self.assertEqual(len(llm.messages), 1)

    async def test_blank_answer_requires_citations_and_support(self):
        from verification import validate_citations
        for text in ("", "   "):
            with self.subTest(text=text):
                value = {**draft(), "text": text}
                with patch("assistant.validate_citations", wraps=validate_citations) as citations:
                    response, llm = await self.chat(value)
                citations.assert_called_once()
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["status"], "answer")
                self.assertEqual(data["text"], value["claims"][0]["text"])
                self.assertEqual(data["claims"][0]["fragment_ids"], [SOURCE.fragment_id])
                self.assertEqual(len(llm.messages), 2)
                checked = json.loads(llm.messages[1][1]["content"])
                self.assertEqual(checked["draft"]["claims"], value["claims"])

    async def test_blank_answer_without_evidence_or_support_is_rejected(self):
        from verification import insufficient_result
        cases = (
            ({"status": "answer", "text": ""}, True, 1),
            ({"status": "answer", "text": "", "claims": []}, True, 1),
            ({**draft(quote="Выдуманная цитата"), "text": ""}, True, 1),
            ({**draft("invented"), "text": ""}, True, 1),
            ({**draft(), "text": ""}, False, 2),
        )
        for value, supported, calls in cases:
            with self.subTest(value=value, supported=supported):
                response, llm = await self.chat(value, supported=supported)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), insufficient_result().model_dump())
                self.assertEqual(len(llm.messages), calls)

    async def test_missing_null_and_wrong_text_types_remain_format_errors(self):
        for status in ("answer", "clarification", "insufficient"):
            for field in ({}, *({"text": t} for t in (None, 123, 1.5, True, [], {}))):
                with self.subTest(status=status, field=field):
                    value = {**draft(), "status": status, **field}
                    if not field:
                        value.pop("text")
                    with self.assertRaises(ValidationError):
                        Draft.model_validate(value)
                    response, llm = await self.chat(value)
                    self.assertEqual(response.status_code, 502)
                    self.assertEqual(len(llm.messages), 1)

    async def test_nonanswer_arbitrary_text_and_fake_sources_are_never_published(self):
        unsafe = "ТЕСТ: любое обязательство прекращается через 999 дней."
        for status in ("clarification", "insufficient"):
            with self.subTest(status=status):
                response, llm = await self.chat({**draft("invented"), "status": status,
                                                 "text": unsafe, "clarification_category": "contract"})
                self.assertEqual(response.status_code, 200)
                self.assertNotIn(unsafe, response.text)
                self.assertNotIn("invented", response.text)
                self.assertEqual(response.json()["claims"], [])
                self.assertEqual(response.json()["fragments"], [])
                self.assertEqual(len(llm.messages), 1)


class APITests(unittest.IsolatedAsyncioTestCase):
    async def call(self, app, method, path, **kwargs):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    async def test_chat_contract(self):
        from api import create_app
        for status in ("answer", "clarification", "insufficient"):
            llm = FakeLLM(draft(), {"supported": True}) if status == "answer" else FakeLLM({"status": status, "text": "Тест"})
            response = await self.call(create_app(evidence=FakeEvidence(), llm=llm), "POST", "/api/chat",
                                       json={"question": "Тест", "history": []})
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(set(data), {"status", "text", "claims", "fragments"})
            self.assertEqual(data["status"], status)
            for fragment in data["fragments"]:
                self.assertIsNone(fragment["start"])
                self.assertIsNone(fragment["end"])

    async def test_validation_and_errors(self):
        from api import create_app
        app = create_app()
        self.assertEqual((await self.call(app, "POST", "/api/chat", json={"question": " "})).status_code, 422)
        self.assertEqual((await self.call(app, "POST", "/api/chat", json={"question": "x"})).status_code, 503)
        for error, expected in ((UpstreamError("secret"), 502), (UpstreamTimeout("secret"), 504),
                                (BackendUnavailable("secret"), 503), (TimeoutError("secret"), 504)):
            app = create_app(evidence=FakeEvidence(), llm=FakeLLM(error))
            response = await self.call(app, "POST", "/api/chat", json={"question": "x"})
            self.assertEqual(response.status_code, expected)
            self.assertNotIn("secret", response.text)
        app = create_app(evidence=FakeEvidence(), llm=FakeLLM("bad JSON"))
        self.assertEqual((await self.call(app, "POST", "/api/chat", json={"question": "x"})).status_code, 502)

    async def test_document_and_html(self):
        from api import create_app
        from pathlib import Path
        from schemas import Document
        async def documents(document_id):
            if document_id == SOURCE.document_id:
                return Document(document_id=document_id, text=SOURCE.text)
            return None
        app = create_app(documents=documents)
        response = await self.call(app, "GET", "/api/documents/test-doc")
        self.assertEqual(response.json(), {"document_id": "test-doc", "text": SOURCE.text})
        self.assertEqual(response.status_code, 200)
        self.assertEqual((await self.call(app, "GET", "/api/documents/unknown")).status_code, 404)
        response = await self.call(app, "GET", "/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, (Path(__file__).parent / "web/static/index.html").read_bytes())

    async def test_document_errors(self):
        from api import create_app
        from schemas import Document
        for error, expected in ((BackendUnavailable(), 503), (UpstreamError(), 502), (UpstreamTimeout(), 504)):
            async def documents(document_id):
                raise error
            response = await self.call(create_app(documents=documents), "GET", "/api/documents/x")
            self.assertEqual(response.status_code, expected)
        async def wrong_document(document_id):
            return Document(document_id="other", text="Тест")
        self.assertEqual((await self.call(create_app(documents=wrong_document), "GET", "/api/documents/x")).status_code, 502)

    async def test_whole_request_deadline(self):
        import asyncio
        from api import create_app
        async def blocked(request):
            await asyncio.Event().wait()
        settings = Settings(request_timeout=0.02, llm_timeout=0.01)
        response = await self.call(create_app(evidence=blocked, settings=settings), "POST", "/api/chat", json={"question": "x"})
        self.assertEqual(response.status_code, 504)


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_settings(self):
        with self.assertRaises(BackendUnavailable):
            await OpenAICompatibleClient(Settings()).complete([])

    async def test_deepseek_thinking_only_for_official_api(self):
        real_client = httpx.AsyncClient
        for base_url, expected in (
            ("https://api.deepseek.com", True),
            ("https://api.deepseek.com/v1", True),
            ("https://other.invalid/v1", False),
            ("https://api.deepseek.com.other.invalid", False),
            ("http://api.deepseek.com", False),
        ):
            calls = []

            def handler(request):
                calls.append(request)
                return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

            settings = Settings(llm_base_url=base_url, llm_api_key="test-only", llm_model="deepseek-flash",
                                retry_attempts=1)
            with self.subTest(base_url=base_url), patch("llm_client.httpx.AsyncClient",
                    side_effect=lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)):
                self.assertEqual(await OpenAICompatibleClient(settings).complete([]), "ok")
                self.assertEqual(len(calls), 1)
                body = json.loads(calls[0].content)
                self.assertEqual(body.get("thinking"), {"type": "disabled"} if expected else None)

    async def test_deepseek_error_handling_with_single_attempt(self):
        real_client = httpx.AsyncClient
        settings = Settings(llm_base_url="https://api.deepseek.com", llm_api_key="test-only",
                            llm_model="deepseek-flash", retry_attempts=1, max_tokens=64)
        for response, error_type in (
            (httpx.Response(200, json={"choices": [{"message": {"content": None,
                                                          "reasoning_content": "synthetic"},
                                                   "finish_reason": "length"}]}), UpstreamError),
            (httpx.Response(429, json={"error": "synthetic"}), UpstreamError),
            (httpx.Response(403, json={"error": "synthetic"}), BackendUnavailable),
        ):
            calls = []

            def handler(request):
                calls.append(request)
                return response

            with self.subTest(status=response.status_code), patch("llm_client.httpx.AsyncClient",
                    side_effect=lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)):
                with self.assertRaises(error_type):
                    await OpenAICompatibleClient(settings).complete([])
                self.assertEqual(len(calls), 1)

    async def test_wire_contract_and_errors(self):
        settings = Settings(llm_base_url="https://synthetic.invalid/v1", llm_api_key="test-only", llm_model="test-model")
        real_client = httpx.AsyncClient
        for mode in ("ok", "http_error", "timeout", "malformed"):
            def handler(request):
                self.assertEqual(str(request.url), "https://synthetic.invalid/v1/chat/completions")
                self.assertEqual(request.headers["Authorization"], "Bearer test-only")
                self.assertEqual(json.loads(request.content)["model"], "test-model")
                if mode == "timeout":
                    raise httpx.ReadTimeout("synthetic", request=request)
                if mode == "http_error":
                    return httpx.Response(429, json={"error": "test"})
                if mode == "malformed":
                    return httpx.Response(200, json={"choices": []})
                return httpx.Response(200, json={"choices": [{"message": {"content": "test-result"}}]})
            with patch("llm_client.httpx.AsyncClient", side_effect=lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)):
                client = OpenAICompatibleClient(settings)
                if mode == "ok":
                    self.assertEqual(await client.complete([]), "test-result")
                else:
                    with self.assertRaises(UpstreamTimeout if mode == "timeout" else UpstreamError):
                        await client.complete([])


class HistoryAndSourceRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_long_history_no_422(self):
        from api import create_app
        history = [{"role": role, "content": f"message-{i}-{role}"}
                   for i in range(70) for role in ("user", "assistant")]
        evidence = FakeEvidence()
        app = create_app(evidence=evidence, llm=FakeLLM({"status": "clarification", "text": "Уточнение"}))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/chat", json={"question": "current", "history": history})
        self.assertEqual(response.status_code, 200)
        actual = evidence.requests[0].user_context
        self.assertLessEqual(len(actual), 24)
        self.assertEqual(actual[-1], "message-69-user")
        self.assertTrue(all(item.endswith("-user") for item in actual))

    async def test_large_messages_budget_and_question_once(self):
        from prompts import answer_messages
        history = [{"role": role, "content": "z" * 30000 + "TAIL"}
                   for _ in range(80) for role in ("user", "assistant")]
        request = ChatRequest(question="UNIQUE_CURRENT", history=history)
        self.assertLessEqual(len(json.dumps([m.model_dump() for m in request.history], ensure_ascii=False)), 12000)
        self.assertTrue(request.history[-1].content.endswith("TAIL"))
        self.assertIn("опущено", request.history[-1].content)
        messages = answer_messages(request, [])
        self.assertEqual(messages[1]["content"].count("UNIQUE_CURRENT"), 1)
        self.assertEqual(ChatRequest.model_validate(request.model_dump()), request)
        self.assertEqual(ChatRequest(question="another").history, [])

    async def test_exact_source_roundtrip_and_nonblank(self):
        from schemas import Document
        text = " \t\r\nМаркер 😀\nвнутренние  пробелы\r\n "
        fragment = Fragment(**{**SOURCE.model_dump(), "text": text})
        document = Document(document_id="d", text=text)
        self.assertEqual(fragment.text, text)
        self.assertEqual(document.text, text)
        self.assertEqual(Fragment.model_validate_json(fragment.model_dump_json()).text, text)
        self.assertIsNone(fragment.start)
        self.assertIsNone(fragment.end)
        for field in ("fragment_id", "document_id", "document_title", "text"):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                Fragment(**{**SOURCE.model_dump(), field: " \t\n"})
        with self.assertRaises(ValidationError):
            Document(document_id="d", text=" \n")

    async def test_search_bridge_has_no_assistant_history(self):
        from dependencies import prepare_search_query
        request = ChatRequest(question="current", history=[
            {"role": "user", "content": "context"},
            {"role": "assistant", "content": "untrusted assertion"}])
        query = prepare_search_query(request)
        self.assertEqual(query.query, "current")
        self.assertEqual(query.user_context, ("context",))
        self.assertEqual(query.filters, ())
        self.assertFalse(hasattr(query, "history"))


class SearchQueryIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_assistant_passes_search_query_not_chat_request(self):
        evidence = FakeEvidence([])
        llm = FakeLLM({"status": "insufficient", "text": "Нет оснований"})
        request = ChatRequest(question="current-marker", history=[
            {"role": "user", "content": "previous-user-marker"},
            {"role": "assistant", "content": "assistant-marker"}])
        await Assistant(evidence, llm, Settings()).answer(request)
        self.assertEqual(len(evidence.requests), 1)
        query = evidence.requests[0]
        self.assertIsInstance(query, SearchQuery)
        self.assertNotIsInstance(query, ChatRequest)
        self.assertEqual(query.query, "current-marker")
        self.assertEqual(query.user_context, ("previous-user-marker",))
        self.assertEqual(query.filters, ())
        self.assertNotIn("assistant-marker", repr(query))
        self.assertEqual(repr(query).count("current-marker"), 1)
        # Full bounded history still reaches the LLM for conversation understanding.
        payload = json.loads(llm.messages[0][1]["content"])
        self.assertEqual(payload["history"][1]["content"], "assistant-marker")

    async def test_separate_calls_do_not_share_context(self):
        evidence = FakeEvidence([])
        llm = FakeLLM(*[{"status": "insufficient", "text": "Нет оснований"}] * 2)
        assistant = Assistant(evidence, llm, Settings())
        await assistant.answer(ChatRequest(question="first", history=[{"role": "user", "content": "first-context"}]))
        await assistant.answer(ChatRequest(question="second"))
        self.assertEqual(evidence.requests[0].user_context, ("first-context",))
        self.assertEqual(evidence.requests[1].user_context, ())


class AdditionalLLMFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_draft_and_support_structures(self):
        for malformed in ({"status": "unknown", "text": "x"},
                          {"status": "answer", "text": "x", "claims": [{"text": "x"}]}):
            with self.subTest(draft=malformed), self.assertRaises(UpstreamError):
                await Assistant(FakeEvidence(), FakeLLM(malformed), Settings()).answer(ChatRequest(question="x"))
        for malformed in ("not JSON", {}, {"supported": []}, {"supported": True, "extra": "x"}):
            with self.subTest(support=malformed), self.assertRaises(UpstreamError):
                await Assistant(FakeEvidence(), FakeLLM(draft(), malformed), Settings()).answer(ChatRequest(question="x"))

    async def test_403_and_invalid_http_json(self):
        real_client = httpx.AsyncClient
        settings = Settings(llm_base_url="https://openrouter.ai/api/v1", llm_model="env-model", llm_api_key="test-only")
        for mode in ("403", "invalid-json"):
            calls = []
            def handler(request):
                calls.append(request)
                self.assertEqual(str(request.url), "https://openrouter.ai/api/v1/chat/completions")
                return httpx.Response(403 if mode == "403" else 200, text="not-json")
            with self.subTest(mode=mode), patch("llm_client.httpx.AsyncClient",
                    side_effect=lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)):
                with self.assertRaises(BackendUnavailable if mode == "403" else UpstreamError):
                    await OpenAICompatibleClient(settings).complete([])
                self.assertEqual(len(calls), 1)

    async def test_missing_key_with_other_settings_present(self):
        settings = Settings(llm_base_url="https://example.invalid/v1", llm_model="env-model")
        with patch("llm_client.httpx.AsyncClient") as http_client:
            with self.assertRaises(BackendUnavailable):
                await OpenAICompatibleClient(settings).complete([])
            http_client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
