"""gRPC-сервер inference. Запуск:  uv run -m src.inference.server

Порт открывается только после полной загрузки модели, адаптера и индекса,
поэтому "порт слушает" == "готов принимать запросы".
"""

import logging
import signal
import threading
from concurrent import futures

import grpc

from src.inference import inference_pb2 as pb
from src.inference import inference_pb2_grpc as pb_grpc
from src.inference.config import settings
from src.inference.engine import GenerationResult, InferenceEngine

log = logging.getLogger("inference")

MAX_MESSAGES = 200
MAX_MESSAGE_CHARS = 32_000


class InferenceService(pb_grpc.InferenceServicer):
    def __init__(self, engine: InferenceEngine):
        self.engine = engine

    def Chat(self, request, context):
        turns = [(m.role, m.content) for m in request.messages]
        last = turns[-1][1] if turns else ""
        log.info("Chat request from %s: messages=%d, last=%r", context.peer(), len(turns), last[:80])
        err = self._validate(turns)
        if err:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, err)

        cancelled = threading.Event()
        context.add_callback(cancelled.set)

        result = GenerationResult()
        try:
            for piece in self.engine.stream(turns, cancelled, result):
                yield pb.ChatResponse(token=piece)
            if cancelled.is_set():
                return
            yield pb.ChatResponse(done=True, finish_reason=result.finish_reason)
        except Exception as e:
            log.exception("generation failed")
            context.abort(grpc.StatusCode.INTERNAL, f"generation failed: {type(e).__name__}")

    @staticmethod
    def _validate(turns) -> str:
        if not turns:
            return "messages is empty"
        if len(turns) > MAX_MESSAGES:
            return f"too many messages (max {MAX_MESSAGES})"
        if turns[-1][0] != "user":
            return "last message must have role 'user'"
        for role, content in turns:
            if role not in ("user", "assistant", "system"):
                return f"unknown role {role!r}"
            if len(content) > MAX_MESSAGE_CHARS:
                return "message too long"
        if not turns[-1][1].strip():
            return "last user message is empty"
        return ""


def serve() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    engine = InferenceEngine(settings)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=settings.grpc_workers))
    pb_grpc.add_InferenceServicer_to_server(InferenceService(engine), server)
    addr = f"{settings.grpc_host}:{settings.grpc_port}"
    if server.add_insecure_port(addr) == 0:
        raise RuntimeError(f"cannot bind {addr}")
    server.start()
    log.info("inference gRPC ready on %s (model=%s, lora=%s, rag=%s)",
             addr, settings.base_model, settings.lora_path or "-", settings.rag_enabled)

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    stop.wait()
    log.info("shutting down...")
    server.stop(grace=10).wait()


if __name__ == "__main__":
    serve()
