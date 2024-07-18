# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from hsmlib import hsm_srv_pb2 as hsm__srv__pb2


class HSMServiceStub(object):
    """Service definition for the HSM library functions
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Load = channel.unary_unary(
                '/hsm.HSMService/Load',
                request_serializer=hsm__srv__pb2.LoadRequest.SerializeToString,
                response_deserializer=hsm__srv__pb2.LoadResponse.FromString,
                )
        self.Create = channel.unary_unary(
                '/hsm.HSMService/Create',
                request_serializer=hsm__srv__pb2.CreateRequest.SerializeToString,
                response_deserializer=hsm__srv__pb2.CreateResponse.FromString,
                )
        self.Add = channel.unary_unary(
                '/hsm.HSMService/Add',
                request_serializer=hsm__srv__pb2.AddRequest.SerializeToString,
                response_deserializer=hsm__srv__pb2.AddResponse.FromString,
                )
        self.Build = channel.unary_unary(
                '/hsm.HSMService/Build',
                request_serializer=hsm__srv__pb2.BuildRequest.SerializeToString,
                response_deserializer=hsm__srv__pb2.BuildResponse.FromString,
                )
        self.Close = channel.unary_unary(
                '/hsm.HSMService/Close',
                request_serializer=hsm__srv__pb2.CloseRequest.SerializeToString,
                response_deserializer=hsm__srv__pb2.CloseResponse.FromString,
                )
        self.Query = channel.unary_unary(
                '/hsm.HSMService/Query',
                request_serializer=hsm__srv__pb2.QueryRequest.SerializeToString,
                response_deserializer=hsm__srv__pb2.QueryResponse.FromString,
                )
        self.Status = channel.unary_unary(
                '/hsm.HSMService/Status',
                request_serializer=hsm__srv__pb2.StatusRequest.SerializeToString,
                response_deserializer=hsm__srv__pb2.StatusResponse.FromString,
                )


class HSMServiceServicer(object):
    """Service definition for the HSM library functions
    """

    def Load(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Create(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Add(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Build(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Close(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Query(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Status(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_HSMServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Load': grpc.unary_unary_rpc_method_handler(
                    servicer.Load,
                    request_deserializer=hsm__srv__pb2.LoadRequest.FromString,
                    response_serializer=hsm__srv__pb2.LoadResponse.SerializeToString,
            ),
            'Create': grpc.unary_unary_rpc_method_handler(
                    servicer.Create,
                    request_deserializer=hsm__srv__pb2.CreateRequest.FromString,
                    response_serializer=hsm__srv__pb2.CreateResponse.SerializeToString,
            ),
            'Add': grpc.unary_unary_rpc_method_handler(
                    servicer.Add,
                    request_deserializer=hsm__srv__pb2.AddRequest.FromString,
                    response_serializer=hsm__srv__pb2.AddResponse.SerializeToString,
            ),
            'Build': grpc.unary_unary_rpc_method_handler(
                    servicer.Build,
                    request_deserializer=hsm__srv__pb2.BuildRequest.FromString,
                    response_serializer=hsm__srv__pb2.BuildResponse.SerializeToString,
            ),
            'Close': grpc.unary_unary_rpc_method_handler(
                    servicer.Close,
                    request_deserializer=hsm__srv__pb2.CloseRequest.FromString,
                    response_serializer=hsm__srv__pb2.CloseResponse.SerializeToString,
            ),
            'Query': grpc.unary_unary_rpc_method_handler(
                    servicer.Query,
                    request_deserializer=hsm__srv__pb2.QueryRequest.FromString,
                    response_serializer=hsm__srv__pb2.QueryResponse.SerializeToString,
            ),
            'Status': grpc.unary_unary_rpc_method_handler(
                    servicer.Status,
                    request_deserializer=hsm__srv__pb2.StatusRequest.FromString,
                    response_serializer=hsm__srv__pb2.StatusResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'hsm.HSMService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class HSMService(object):
    """Service definition for the HSM library functions
    """

    @staticmethod
    def Load(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/hsm.HSMService/Load',
            hsm__srv__pb2.LoadRequest.SerializeToString,
            hsm__srv__pb2.LoadResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Create(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/hsm.HSMService/Create',
            hsm__srv__pb2.CreateRequest.SerializeToString,
            hsm__srv__pb2.CreateResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Add(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/hsm.HSMService/Add',
            hsm__srv__pb2.AddRequest.SerializeToString,
            hsm__srv__pb2.AddResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Build(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/hsm.HSMService/Build',
            hsm__srv__pb2.BuildRequest.SerializeToString,
            hsm__srv__pb2.BuildResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Close(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/hsm.HSMService/Close',
            hsm__srv__pb2.CloseRequest.SerializeToString,
            hsm__srv__pb2.CloseResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Query(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/hsm.HSMService/Query',
            hsm__srv__pb2.QueryRequest.SerializeToString,
            hsm__srv__pb2.QueryResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Status(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/hsm.HSMService/Status',
            hsm__srv__pb2.StatusRequest.SerializeToString,
            hsm__srv__pb2.StatusResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)