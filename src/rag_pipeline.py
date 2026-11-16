"""
RAG Inference Pipeline for Avito Onboarding Assistant.
Combines Vector Retrieval (FAISS + E5) with fine-tuned Qwen2.5-7B LoRA adapter.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from src.build_index import VectorKnowledgeBase

DEFAULT_SYSTEM_PROMPT = (
    "Ты — виртуальный ассистент по адаптации новых сотрудников Авито. "
    "Отвечай вежливо, структурированно и строго на русском языке. "
    "Категорически запрещено использовать китайский язык и иностранные иероглифы. "
    "Если информации нет в базе знаний компании, прямо скажи об этом и посоветуй обратиться к бадди или руководителю."
)


class AvitoRAGPipeline:
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        adapter_dir: str = "adapters/qwen_lora",
        index_dir: str = "data/index",
        embedding_model_name: str = "intfloat/multilingual-e5-small",
        load_in_4bit: bool = True,
        force_cpu: bool = False,
    ):
        """
        Initializes the retriever and loads the fine-tuned LLM.
        """
        self.base_model_name = base_model_name
        self.adapter_dir = Path(adapter_dir)
        self.index_dir = Path(index_dir)
        self.device = "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"

        print(f"[Init] Target execution device: {self.device.upper()}")

        # 1. Initialize Vector Knowledge Base
        print(f"[Init] Loading vector knowledge base from '{self.index_dir}'...")
        self.kb = VectorKnowledgeBase(
            model_name=embedding_model_name,
            index_dir=str(self.index_dir),
        )
        self.kb.load()
        print(
            f"[Init] Knowledge base ready ({len(self.kb.documents)} documents indexed)."
        )

        # 2. Load Tokenizer
        print(f"[Init] Loading tokenizer for '{base_model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 3. Load Model with optional 4-bit quantization on GPU
        print(f"[Init] Loading model '{base_model_name}'...")
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
        }

        if self.device == "cuda":
            if load_in_4bit:
                try:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    model_kwargs["quantization_config"] = bnb_config
                    model_kwargs["device_map"] = "auto"
                    print("[Init] 4-bit QLoRA quantization enabled on GPU.")
                except Exception as e:
                    print(
                        f"[Init] Warning: Could not configure 4-bit quantization ({e}). Falling back to float16."
                    )
                    model_kwargs["torch_dtype"] = torch.float16
                    model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
        else:
            print("[Init] Running on CPU (quantization disabled).")
            model_kwargs["torch_dtype"] = torch.float32
            model_kwargs["device_map"] = "cpu"

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            **model_kwargs,
        )

        # 4. Attach LoRA Adapter if present
        if (
            self.adapter_dir.exists()
            and (self.adapter_dir / "adapter_model.safetensors").exists()
        ):
            print(f"[Init] Attaching LoRA adapter from '{self.adapter_dir}'...")
            self.model = PeftModel.from_pretrained(base_model, str(self.adapter_dir))
        else:
            print(
                f"[Init] Notice: LoRA adapter not found at '{self.adapter_dir}'. Running base model directly."
            )
            self.model = base_model

        self.model.eval()
        print("[Init] RAG Pipeline successfully initialized and ready.\n")

    def retrieve(
        self, query: str, top_k: int = 3
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieves top_k context passages from the knowledge base.
        """
        return self.kb.search(query, top_k=top_k)

    def build_prompt(
        self,
        query: str,
        context_docs: Optional[List[Dict[str, Any]]] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> str:
        """
        Builds ChatML formatted prompt containing system instructions,
        retrieved knowledge context, and user inquiry.
        """
        if context_docs:
            context_blocks = []
            for i, doc in enumerate(context_docs, 1):
                category = f" [{doc['category']}]" if doc.get("category") else ""
                content = doc.get("content", "").strip()
                context_blocks.append(f"Фрагмент #{i}{category}:\n{content}")

            context_text = "\n\n".join(context_blocks)
            system_with_context = (
                f"{system_prompt}\n\n"
                f"Ниже приведена официальная информация из базы знаний компании. "
                f"Используй её для формирования точного и полного ответа сотруднику:\n\n"
                f"{context_text}"
            )
        else:
            system_with_context = system_prompt

        prompt_str = (
            f"<|im_start|>system\n{system_with_context}<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt_str

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> str:
        """
        Generates model response given formatted input prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        prompt_length = input_ids.shape[1]

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if temperature > 0:
            generation_config["do_sample"] = True
            generation_config["temperature"] = temperature
            generation_config["top_p"] = top_p
        else:
            generation_config["do_sample"] = False

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )

        generated_tokens = outputs[0][prompt_length:]
        response = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()
        return response

    def ask(
        self,
        query: str,
        top_k: int = 3,
        use_rag: bool = True,
        return_sources: bool = False,
    ) -> Dict[str, Any]:
        """
        Full end-to-end question answering pipeline.
        """
        retrieved_hits = []
        context_docs = []

        if use_rag:
            retrieved_hits = self.retrieve(query, top_k=top_k)
            context_docs = [doc for doc, score in retrieved_hits]

        prompt = self.build_prompt(query, context_docs if use_rag else None)
        answer = self.generate(prompt)

        result: Dict[str, Any] = {
            "query": query,
            "answer": answer,
        }

        if return_sources:
            result["sources"] = [
                {
                    "content": doc.get("content"),
                    "category": doc.get("category"),
                    "intent_id": doc.get("intent_id"),
                    "score": round(score, 4),
                }
                for doc, score in retrieved_hits
            ]

        return result


def interactive_chat(pipeline: AvitoRAGPipeline, top_k: int = 3, use_rag: bool = True):
    """
    Starts an interactive terminal chat session.
    """
    print("\n" + "=" * 60)
    print(" Ассистент адаптации Авито готов к работе!")
    print(
        f"Режим RAG: {'ВКЛЮЧЕН (top_k=' + str(top_k) + ')' if use_rag else 'ВЫКЛЮЧЕН'}"
    )
    print("Введите вопрос сотрудника (или 'exit'/'quit' для завершения):")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("\n👤 Сотрудник: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Завершение сессии.")
                break

            print(" Поиск информации и генерация ответа...")
            res = pipeline.ask(query, top_k=top_k, use_rag=use_rag, return_sources=True)

            print(f"\n Ассистент:\n{res['answer']}\n")

            if use_rag and res.get("sources"):
                print("📚 Использованные фрагменты базы знаний:")
                for i, src in enumerate(res["sources"], 1):
                    cat = f" [{src['category']}]" if src.get("category") else ""
                    print(
                        f"  {i}. (сходство: {src['score']}){cat}: {src['content'][:100]}..."
                    )

        except KeyboardInterrupt:
            print("\nСессия прервана пользователем.")
            break


def main():
    parser = argparse.ArgumentParser(description="Avito Onboarding RAG Inference")
    parser.add_argument(
        "--query", type=str, default=None, help="Одиночный вопрос для проверки"
    )
    parser.add_argument(
        "--chat", action="store_true", help="Запустить интерактивный консольный чат"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Количество документов из базы знаний (по умолчанию: 3)",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Отключить RAG (чистый ответ модели без контекста)",
    )
    parser.add_argument(
        "--no-4bit", action="store_true", help="Отключить 4-битное квантование"
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Принудительно выполнять на CPU"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Базовая модель",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default="adapters/qwen_lora",
        help="Директория с LoRA-адаптером",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="data/index",
        help="Директория с FAISS-индексом",
    )

    args = parser.parse_args()

    pipeline = AvitoRAGPipeline(
        base_model_name=args.base_model,
        adapter_dir=args.adapter_dir,
        index_dir=args.index_dir,
        load_in_4bit=not args.no_4bit,
        force_cpu=args.cpu,
    )

    if args.query:
        print(f"\nВопрос: {args.query}\n")
        res = pipeline.ask(
            args.query, top_k=args.top_k, use_rag=not args.no_rag, return_sources=True
        )
        print(f"Ответ:\n{res['answer']}\n")
        if res.get("sources"):
            print("Источники:")
            for s in res["sources"]:
                print(f"- ({s['score']}): {s['content'][:120]}...")
    elif args.chat or len(sys.argv) == 1:
        interactive_chat(pipeline, top_k=args.top_k, use_rag=not args.no_rag)


if __name__ == "__main__":
    main()
