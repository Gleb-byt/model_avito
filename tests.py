from src.rag_pipeline import AvitoRAGPipeline

pipeline = AvitoRAGPipeline(
    base_model_name="Qwen/Qwen2.5-7B-Instruct",
    adapter_dir="adapters/qwen_lora",
    index_dir="data/index",
    load_in_4bit=True,
)

result = pipeline.ask(
    query="Что делать если ничего не получается?", top_k=3, return_sources=True
)

print(result["answer"])
for src in result["sources"]:
    print(f"-[{src['score']}] {src['content'][:80]}...")
