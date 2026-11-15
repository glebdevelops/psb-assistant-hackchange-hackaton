# Юридический ассистент ПСБ

## Веб-прототип через Docker

Нужны Docker Desktop (или Docker Engine с Compose) и каталог с `knowledge-base.zip`. Архив подключается к контейнеру только для чтения и не включается в образ. По умолчанию Compose ищет его в `./data/knowledge-base.zip`; путь к другому каталогу можно передать через `PSB_DATA_DIR`.

Пример для PowerShell, если архив лежит в Downloads:

```powershell
$env:PSB_DATA_DIR = 'C:\Users\User\Downloads'
docker compose up --build
```

Откройте `http://localhost:8000/`. Для полных ответов также задайте `LLM_BASE_URL`, `LLM_API_KEY` и `LLM_MODEL` в окружении перед запуском. Адрес прокси должен быть доступен из контейнера: `localhost` внутри контейнера обозначает сам контейнер. Секреты не записывайте в репозиторий или Dockerfile.

Остановка: `docker compose down`. Проверка конфигурации без запуска: `docker compose config --quiet`. Контейнер запускает `uvicorn api:app`; при старте API один раз строит индекс из `DATA_DIR/knowledge-base.zip` и обслуживает чат и просмотр документов. Наружу открыт только локальный порт `127.0.0.1:8000`.

Для автоматического ZIP-сабмита платформа переносит только корневые `.py`-файлы и `requirements.txt`. Docker-файлы и `web/` относятся к финальному прототипу и не нужны `solution.py` при автоматической оценке.
