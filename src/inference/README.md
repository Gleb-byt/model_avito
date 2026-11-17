# inference — gRPC-сервис модели

Стримит токены Qwen2.5-7B-Instruct + LoRA (`adapters/qwen_lora`) с RAG по базе знаний (`data/index`).
Контракт: `proto/inference/v1/inference.proto` (копия из бэкенда `avito-api/api/proto/...`, держать синхронно).

## Запуск

```bash
uv pip install -r requirements.txt
uv run -m src.inference.server
```

Порт открывается **после** загрузки модели (минуты) — «порт слушает» значит «готов».

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `GRPC_PORT` | `50051` | порт |
| `BASE_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | базовая модель |
| `LORA_PATH` | `adapters/qwen_lora` | адаптер; пусто — без него |
| `QUANTIZATION` | `auto` | `auto` = 4bit nf4 на CUDA, без квантизации на CPU |
| `RAG_ENABLED` | `true` | подмешивать найденные пассажи в system-промпт |
| `RAG_TOP_K` / `RAG_MIN_SCORE` | `3` / `0.80` | сколько пассажей и порог сходства |
| `MAX_NEW_TOKENS` | `512` | лимит ответа |
| `MAX_CONCURRENT` | `1` | одновременных генераций (одна GPU → 1) |

## Требования к железу

7B в 4bit: ~6 ГБ VRAM, нужна CUDA (bitsandbytes). На CPU 7B запускается, но очень медленно —
для отладки используйте `BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct LORA_PATH=` (адаптер обучен под 7B и к малой модели не подходит).

## Как это работает

1. `Chat(messages)` → последний `user` — запрос для поиска в FAISS (`multilingual-e5-small`).
2. Промпт: system (`DEFAULT_SYSTEM_PROMT` + найденные пассажи) + история (обрезается по `MAX_INPUT_TOKENS`, старые сообщения выкидываются первыми).
3. `generate()` в отдельном потоке, `TextIteratorStreamer` отдаёт текст кусками → `ChatResponse{token}`.
4. Отмена клиентом (кнопка «стоп», обрыв, таймаут) → `context.add_callback` → `StoppingCriteria` останавливает `generate()`.
5. В конце `ChatResponse{done: true, finish_reason: stop|length}`.

Ошибки: `INVALID_ARGUMENT` — пустой/некорректный запрос; `INTERNAL` — сбой генерации.

## Перегенерация stubs после правки proto

```bash
uv run -m grpc_tools.protoc -I proto/inference/v1 --python_out=src/inference --grpc_python_out=src/inference inference.proto
# затем в src/inference/inference_pb2_grpc.py:  import inference_pb2 as ...  ->  from . import inference_pb2 as ...
```
