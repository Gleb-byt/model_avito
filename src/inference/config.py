"""Настройки inference-сервиса. Всё задаётся переменными окружения (или .env)."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str) -> str:
    return os.getenv(key) or default


@dataclass(frozen=True)
class Settings:
    # gRPC
    grpc_host: str = _env("GRPC_HOST", "0.0.0.0")
    grpc_port: int = int(_env("GRPC_PORT", "50051"))
    grpc_workers: int = int(_env("GRPC_WORKERS", "8"))

    # Модель. LORA_PATH="" -> без адаптера (удобно для отладки на маленькой модели).
    base_model: str = _env("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    lora_path: str = os.getenv("LORA_PATH", "adapters/qwen_lora")
    # auto: 4bit nf4 на CUDA (как при обучении), без квантизации на CPU. Варианты: auto | 4bit | none
    quantization: str = _env("QUANTIZATION", "auto")

    # Генерация
    max_new_tokens: int = int(_env("MAX_NEW_TOKENS", "512"))
    max_input_tokens: int = int(_env("MAX_INPUT_TOKENS", "3072"))
    temperature: float = float(_env("TEMPERATURE", "0.3"))
    top_p: float = float(_env("TOP_P", "0.9"))
    repetition_penalty: float = float(_env("REPETITION_PENALTY", "1.05"))
    # Сколько генераций одновременно (одна GPU -> 1). Остальные ждут в очереди.
    max_concurrent: int = int(_env("MAX_CONCURRENT", "1"))
    # Макс. тишина между токенами, сек: защита от зависшей генерации.
    stream_timeout: float = float(_env("STREAM_TIMEOUT", "120"))

    # RAG
    rag_enabled: bool = _env("RAG_ENABLED", "true").lower() == "true"
    embed_model: str = _env("EMBED_MODEL", "intfloat/multilingual-e5-small")
    index_dir: str = _env("INDEX_DIR", "data/index")
    top_k: int = int(_env("RAG_TOP_K", "3"))
    # Ниже этого косинусного сходства документ в контекст не берём.
    min_score: float = float(_env("RAG_MIN_SCORE", "0.80"))


settings = Settings()
