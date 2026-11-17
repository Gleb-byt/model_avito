import grpc
import warnings

from . import inference_pb2 as inference__pb2

GRPC_GENERATED_VERSION = '1.85.1'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + ' but the generated code in inference_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class InferenceStub:
    def __init__(self, channel):
        self.Chat = channel.unary_stream(
                '/inference.v1.Inference/Chat',
                request_serializer=inference__pb2.ChatRequest.SerializeToString,
                response_deserializer=inference__pb2.ChatResponse.FromString,
                _registered_method=True)


class InferenceServicer:
    def Chat(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_InferenceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Chat': grpc.unary_stream_rpc_method_handler(
                    servicer.Chat,
                    request_deserializer=inference__pb2.ChatRequest.FromString,
                    response_serializer=inference__pb2.ChatResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'inference.v1.Inference', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('inference.v1.Inference', rpc_method_handlers)


class Inference:
    @staticmethod
    def Chat(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(
            request,
            target,
            '/inference.v1.Inference/Chat',
            inference__pb2.ChatRequest.SerializeToString,
            inference__pb2.ChatResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
