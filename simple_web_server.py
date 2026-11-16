# uvicorn simple_web_server:app --port 8000 --host 0.0.0.0

from fastapi import FastAPI
from pydantic import BaseModel
from src.rag_pipeline import AvitoRAGPipeline

app = FastAPI(title="Avito RAG API")

# Загружаем модель один раз при старте приложения
pipeline = AvitoRAGPipeline(
    base_model_name="Qwen/Qwen2.5-7B-Instruct",
    adapter_dir="adapters/qwen_lora",
    index_dir="data/index",
    load_in_4bit=True,
)


class Question(BaseModel):
    text: str


@app.get("/")
def read_root():
    return {"status": "online", "message": "Avito RAG API is running!", "docs": "/docs"}


@app.post("/get_answer")
def get_answer(req: Question):
    print(f"got request: {req.text}")

    result = pipeline.ask(query=req.text, top_k=3, return_sources=False)

    return {"name": "Qwen/Qwen2.5-7B-Instruct", "content": result}
