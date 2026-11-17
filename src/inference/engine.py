import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from src.build_index import VectorKnowledgeBase
from src.data_loader import DEFAULT_SYSTEM_PROMPT
from src.inference.config import Settings

Turn = Tuple[str, str]  # (role, content)


class _CancelCriteria(StoppingCriteria):
    def __init__(self, *events: threading.Event):
        self.events = events

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return any(e.is_set() for e in self.events)


@dataclass
class GenerationResult:
    """Заполняется по завершении стрима."""

    finish_reason: str = "stop"


class InferenceEngine:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self._slots = threading.BoundedSemaphore(cfg.max_concurrent)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
        self.model = self._load_model()
        self.kb = self._load_kb()

    def _load_model(self):
        cfg = self.cfg
        use_4bit = cfg.quantization == "4bit" or (
            cfg.quantization == "auto" and self.device == "cuda"
        )
        kwargs = {}
        if use_4bit:
            if self.device != "cuda":
                raise RuntimeError("QUANTIZATION=4bit требует CUDA")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = "auto"
        else:
            kwargs["dtype"] = torch.float16 if self.device == "cuda" else torch.bfloat16

        print(f"[engine] loading {cfg.base_model} (device={self.device}, 4bit={use_4bit})", flush=True)
        model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **kwargs)

        if cfg.lora_path:
            if not Path(cfg.lora_path, "adapter_config.json").exists():
                raise FileNotFoundError(f"LoRA адаптер не найден: {cfg.lora_path}")
            print(f"[engine] applying LoRA {cfg.lora_path}", flush=True)
            model = PeftModel.from_pretrained(model, cfg.lora_path)

        if not use_4bit:
            model.to(self.device)
        model.eval()
        return model

    def _load_kb(self) -> Optional[VectorKnowledgeBase]:
        if not self.cfg.rag_enabled:
            return None
        print(f"[engine] loading RAG index from {self.cfg.index_dir}", flush=True)
        kb = VectorKnowledgeBase(model_name=self.cfg.embed_model, index_dir=self.cfg.index_dir)
        kb.load()
        return kb


    def retrieve(self, query: str) -> List[str]:
        if self.kb is None or not query.strip():
            return []
        hits = self.kb.search(query, top_k=self.cfg.top_k)
        return [doc["content"] for doc, score in hits if score >= self.cfg.min_score]

    def build_input_ids(self, turns: Sequence[Turn]):
        """system (+RAG-контекст) + история, обрезанная по бюджету токенов."""
        last_user = next((c for r, c in reversed(turns) if r == "user"), "")
        passages = self.retrieve(last_user)

        system = DEFAULT_SYSTEM_PROMPT
        if passages:
            ctx = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
            system += (
                "\n\nИспользуй для ответа следующую информацию из базы знаний. "
                "Если в ней нет ответа, честно скажи об этом.\n\n" + ctx
            )

        history = [{"role": r, "content": c} for r, c in turns if r in ("user", "assistant")]

        def encode(h):
            text = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, *h],
                tokenize=False,
                add_generation_prompt=True,
            )
            return self.tokenizer(text, return_tensors="pt", add_special_tokens=False)

        enc = encode(history)
        while enc["input_ids"].shape[1] > self.cfg.max_input_tokens and len(history) > 1:
            history = history[1:]
            enc = encode(history)
        return enc


    def stream(
        self,
        turns: Sequence[Turn],
        cancelled: threading.Event,
        result: GenerationResult,
        max_new_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """Отдаёт текст по мере генерации. Прерывается, когда cancelled выставлен."""
        while not self._slots.acquire(timeout=0.2):
            if cancelled.is_set():
                result.finish_reason = "cancelled"
                return
        try:
            yield from self._generate(turns, cancelled, result, max_new_tokens)
        finally:
            self._slots.release()

    def _generate(self, turns, cancelled, result, max_new_tokens) -> Iterator[str]:
        cfg = self.cfg
        limit = max_new_tokens or cfg.max_new_tokens
        enc = self.build_input_ids(turns)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=cfg.stream_timeout,
        )
        out: dict = {}
        stop = threading.Event()

        def run() -> None:
            try:
                with torch.inference_mode():
                    out["ids"] = self.model.generate(
                        **enc,
                        streamer=streamer,
                        max_new_tokens=limit,
                        do_sample=cfg.temperature > 0,
                        temperature=cfg.temperature if cfg.temperature > 0 else None,
                        top_p=cfg.top_p,
                        repetition_penalty=cfg.repetition_penalty,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                        stopping_criteria=StoppingCriteriaList([_CancelCriteria(cancelled, stop)]),
                    )
            except BaseException as e:
                out["error"] = e
                streamer.on_finalized_text("", stream_end=True)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        try:
            for piece in streamer:
                if cancelled.is_set():
                    break
                if piece:
                    yield piece
        finally:
            stop.set()
            worker.join()

        if "error" in out:
            raise out["error"]
        if cancelled.is_set():
            result.finish_reason = "cancelled"
        elif out.get("ids") is not None and out["ids"].shape[1] - prompt_len >= limit:
            result.finish_reason = "length"
        else:
            result.finish_reason = "stop"
