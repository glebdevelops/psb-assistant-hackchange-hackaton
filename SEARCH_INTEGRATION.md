# Интеграция поиска с юридическим ассистентом ПСБ

Статус: **готов к интеграции**. Реальный поиск уже подключён к `solution.py` через `dependencies.initialize_search()` и к обычному FastAPI-приложению через startup в `api.py`. Этот документ описывает фактический код, а не предполагаемый интерфейс.

## Файлы и граница ответственности

| Корневой файл | Назначение |
|---|---|
| `search_corpus.py` | Однократное чтение ZIP, полный исходный текст, дословные фрагменты, стабильные ID и координаты |
| `search_index.py` | Неизменяемый обратный индекс и оценка BM25 с точными идентификаторами |
| `search_service.py` | Синхронный поиск, фильтры, ранжирование и получение полного документа |
| `search_adapter.py` | Асинхронные `EvidenceProvider` и `DocumentProvider` для backend-моделей |
| `dependencies.py` | Backend-контракт и штатная фабрика `initialize_search` |
| `api.py` | Однократная загрузка поиска при запуске обычного API |

Все production-модули находятся в корне. `solution.py` импортирует только корневые `.py`; ему не нужны `web/`, тесты или другие вложенные файлы. Новых зависимостей нет, `requirements.txt` не менялся. База знаний берётся из входного `knowledge-base.zip`; индекс строится в памяти при запуске и не кладётся в ZIP-сабмит.

Поиск **не формирует юридический ответ** и не обращается к LLM. Он возвращает только реальные фрагменты предоставленного архива. Генерация и проверка выводов остаются в `assistant.py` и `verification.py`.

## Публичные сигнатуры

```python
# search_corpus.py
load_corpus(archive_path: Path, *, chunk_chars: int = 1800) -> Corpus
Corpus.get_document(document_id: str) -> CorpusDocument | None

# search_index.py
SearchIndex.build(corpus: Corpus) -> SearchIndex

# search_service.py
load_search_service(archive_path: Path, *, chunk_chars: int = 1800) -> SearchService
SearchService.search(
    query: str, *, context: tuple[str, ...] = (),
    filters: tuple[tuple[str, str], ...] = (), top_k: int = 10,
) -> list[SearchHit]
SearchService.get_document(document_id: str) -> CorpusDocument | None

# search_adapter.py
create_search_components(archive_path: Path, settings: Settings) -> SearchComponents  # async
bind_search_service(service: SearchService, settings: Settings) -> SearchComponents

# dependencies.py — уже подключённая backend-фабрика
initialize_search(archive_path: Path, settings: Settings) -> SearchComponents  # async
```

`SearchHit.fragment` содержит `CorpusFragment`; `SearchHit.score` служит только для порядка выдачи. `CorpusDocument.text` — полный текст именно той версии файла, из которой взят фрагмент. `Corpus.get_document()` и `SearchService.get_document()` возвращают `None` только для неизвестного ID.

## Соответствие backend-контракту

| Backend | Поисковый модуль | Что делает адаптер |
|---|---|---|
| `SearchFactory(Path, Settings)` | `load_search_service(Path)` | `dependencies.initialize_search()` вызывает `create_search_components()` один раз |
| `SearchComponents` | Один `SearchService` | Возвращает замыкания `evidence` и `documents`, использующие общий индекс |
| `SearchQuery.query` | `search(query=...)` | Передаёт текущий вопрос как есть |
| `SearchQuery.user_context` | `search(context=...)` | Передаёт только предыдущие сообщения пользователя с меньшим весом |
| `SearchQuery.filters` | `search(filters=...)` | Передаёт только подтверждённые пары без автоматических догадок |
| `Settings.top_k` | `search(top_k=...)` | Ограничивает количество результатов |
| `list[Fragment]` | `list[SearchHit]` | Проверяет дословное совпадение с документом, создаёт Pydantic `Fragment`, отбрасывает `score` |
| `Document | None` | `get_document(document_id)` | Преобразует полный исходный текст в Pydantic `Document` |

Сейчас допустимы точные фильтры `document_id`, `kind` (`company`/`law`), `revision`, `source_name`. Неподдерживаемый фильтр вызывает `ValueError` в сервисе и `UpstreamError` на границе backend. Автоматическое извлечение фильтров из вопроса не реализовано. Дату события сервис использует только как подсказку для ранжирования двух внутренних регламентов; он **не устанавливает юридическую применимость** и не отбрасывает другую редакцию.

### Дословность и координаты

Для внутреннего `CorpusFragment` верно:

```python
fragment.text == document.text[fragment.start:fragment.end]
```

`start/end` — индексы кодовых точек Unicode в Python, отсчёт от начала полного `CorpusDocument.text`, `end` исключительный. Исходные пробелы, переносы строк и Unicode после определения границ не нормализуются. Текущий backend `Fragment` допускает для `start/end` только `None`, поэтому адаптер передаёт `None` и проверяет текст по внутренним координатам до преобразования. Повторяющиеся одинаковые цитаты могут остаться неоднозначными для подсветки во frontend до согласованного расширения схемы координат.

`document_id` зависит от имени файла внутри ZIP и его точных байтов; `fragment_id` — от `document_id` и координат. Порядок файлов ZIP не влияет на ID. Изменённый текст получает новый ID, что отделяет редакции. У законов `revision` сейчас содержит дату выгрузки, а не доказанный период действия; `valid_from/valid_to` остаются пустыми, пока нет надёжного основания их заполнить.

## Пример реального вызова

```python
from pathlib import Path
from config import Settings
from dependencies import SearchQuery, initialize_search

components = await initialize_search(Path("data/knowledge-base.zip"), Settings(top_k=10))
fragments = await components.evidence(SearchQuery(
    query="Какой срок оплаты по договору П-17/2024?",
    user_context=("Речь о договоре с Альфа-Трейд",),
    filters=(),
))
if fragments:
    document = await components.documents(fragments[0].document_id)
    assert document is not None
    assert fragments[0].text in document.text
```

Этот вызов уже выполняется в штатном CSV-пути `solution.run()`. Для обычного API `api.app` вызывает ту же фабрику при FastAPI startup: путь — `DATA_DIR/knowledge-base.zip`, по умолчанию `data/knowledge-base.zip`. Для явного пути доступен `api.create_app(archive_path=Path(...))`. Внедрение готовых `evidence`/`documents` через `create_app()` сохраняется для тестов.

## Ошибки и параллельность

- Успешный поиск без кандидатов: `[]`. Неизвестный `document_id`: `None`.
- Отсутствующий ZIP при инициализации: `BackendUnavailable`.
- Повреждённый архив, неверная кодировка/структура или некорректный фильтр: `UpstreamError` на границе адаптера.
- Тайм-аут поиска/инициализации: `UpstreamTimeout` либо внешний `TimeoutError` из ограничений backend. Ошибки не заменяются пустой выдачей.
- Индекс и корпус после построения неизменяемы; `SearchService.search()` использует локальное состояние. Одновременные запросы дают те же ID и порядок, что последовательные.
- Загрузка и CPU-поиск синхронны внутри сервиса. Адаптер вызывает их через `asyncio.to_thread()`, чтобы не блокировать event loop. `MAX_WORKERS` в CSV по умолчанию равен 2; индекс переиспользуется всеми вопросами одного запуска.

Строка служебной записки, обращённая к ИИ-ассистенту, остаётся в полном исходном документе, но соответствующий `untrusted_instruction` не индексируется и не попадает в поисковую выдачу. Редакционные примечания индексируются с пониженным весом для обычных вопросов. Любой текст архива и пользовательского контекста остаётся недоверенными данными для backend-промптов.

## Проверки и воспроизведение

Новые тесты находятся в корне: `test_search.py` и `test_search_integreation.py` (написание второго имени сохранено по запросу). Они содержат отдельно обозначенные небольшие синтетические проверки ошибок и проверку реального архива. Для повторения вне рабочей машины задайте `PSB_TEST_ARCHIVE` полным путём к `knowledge-base.zip`; альтернативно тесты ищут его в `DATA_DIR` и `~/Downloads`. При отсутствии архива реальные проверки пропускаются, синтетические продолжают выполняться.

```powershell
$env:PSB_TEST_ARCHIVE = 'C:\path\to\knowledge-base.zip'
python -B -m unittest -v test_search test_search_integreation
```

На предоставленном архиве: 21 документ, 1032 фрагмента, из них 1031 индексируется. Новые 15 тестов проходят. Проверены похожие контрагенты, допсоглашение и исходный договор, две редакции регламента, письмо комплаенс-службы и п. 5.4, статья 164 НК РФ, точность фрагментов, ID, пустая выдача, ошибки, получение документа, параллельность, API и CSV-путь. Локальный замер построения индекса — около 0,09 с; поиск — порядка нескольких миллисекунд. Это замер на данном архиве, не гарантия для иных входных данных.

В старом `test_solution.py` остаётся тест `test_missing_real_search_does_not_create_output`, который ожидает прежний stub `integration is unfinished`. После реального подключения он падает: подаёт некорректный ZIP и получает `UpstreamError`, как и должен. Сам тест не менялся, поскольку он не входит в разрешённый список файлов этого этапа. При обновлении набора тестов нужно заменить его ожидание на проверку ошибки повреждённого архива.

## Известные пределы

- Ранжирование лексическое (BM25, номера, заголовки, небольшие словарные расширения). Сложное семантическое перефразирование может потребовать другой стратегии поиска.
- Сервис не восстанавливает отсутствующие исторические редакции законов и не решает самостоятельно, какое правовое положение применимо к событию.
- Редакционные пометки выделяются эвристически; `source_text` само по себе не означает действующую норму.
- Полный прогон с настоящим LLM-прокси и юридическое качество ответов не проверялись. Тестовый LLM проверяет формат цепочки и цитирование, а не юридическую истинность.
