import argparse
import os
from builtins import range
from functools import partial
import struct

import grpc
import numpy as np
import sys

from tensorrtserver.api import ProtocolType, ServerStatusContext
from tensorrtserver.api import api_pb2
from tensorrtserver.api import grpc_service_pb2
from tensorrtserver.api import grpc_service_pb2_grpc
import tensorrtserver.api.model_config_pb2 as model_config


DTYPE = {
    model_config.TYPE_BOOL: np.bool,
    model_config.TYPE_FP16: np.float16,
    model_config.TYPE_FP32: np.float32,
    model_config.TYPE_FP64: np.float64,
    model_config.TYPE_INT8: np.int8,
    model_config.TYPE_INT32: np.int32,
    model_config.TYPE_INT64: np.int64,
    model_config.TYPE_UINT8: np.uint8,
    model_config.TYPE_UINT16: np.uint16,
    model_config.TYPE_UINT32: np.uint32,
    model_config.TYPE_STRING: np.str
}


def _parse_model(url, model_name, verbose=False):
    protocol = ProtocolType.from_str('gRPC')
    ctx = ServerStatusContext(url, protocol, model_name, verbose)
    server_status = ctx.get_server_status()

    if model_name not in server_status.model_status:
        raise Exception("unable to get status for '" + model_name + "'")

    status = server_status.model_status[model_name]
    config = status.config

    return config


class Inference(object):

    def __init__(self, url, model_name, model_version):
        model = _parse_model(url, model_name)
        self.url = url
        self.model = model
        self.model_name = model_name
        self.model_version = model_version

    def _to_bytes(self, input_value):
        input_value = np.array([input_value], dtype=bytes)
        flattened = bytes()
        for obj in np.nditer(input_value, flags=["refs_ok"], order='C'):
            # If directly passing bytes to STRING type,
            # don't convert it to str as Python will encode the
            # bytes which may distort the meaning
            if obj.dtype.type == np.bytes_:
                s = bytes(obj)
            else:
                s = str(obj).encode('utf-8')
            flattened += struct.pack("<I", len(s))
            flattened += s

        input_value = np.asarray(flattened)
        byte_size = input_value.itemsize * input_value.size
        return input_value.tobytes(), byte_size

    def build_request(self, input):
        request = grpc_service_pb2.InferRequest()
        request.model_name = self.model_name
        request.model_version = self.model_version
        batch_size = self.model.input[0].dims[0]
        request.meta_data.batch_size = batch_size

        for output in self.model.output:
            request.meta_data.output.add(name=output.name)
            # output_message = api_pb2.InferRequestHeader.Output()
            # output_message.name = output.name
            # request.meta_data.output.extend([output_message])

        del request.raw_input[:]

        for name, value in input.items():
            if type(value) in [bytes, str]:
                input_bytes, byte_size = self._to_bytes(value)
                request.meta_data.input.add(
                    name=name, dims=[1],
                    batch_byte_size=byte_size
                )
                request.raw_input.append(input_bytes)

            elif type(value) is np.ndarray:
                input_bytes = value.tobytes()
                batch_byte_size = len(input_bytes)
                request.meta_data.input.add(
                    name=name, dims=list(value.shape),
                    batch_byte_size=batch_byte_size
                )
                request.raw_input.append(input_bytes)

        #print(request)
        return request

    def get_status(self, grpc_stub):
        request = grpc_service_pb2.StatusRequest(model_name=self.model_name)
        status = grpc_stub.Status(request)
        return status

    def run(self, input):
        channel = grpc.insecure_channel(self.url)
        grpc_stub = grpc_service_pb2_grpc.GRPCServiceStub(channel)

        request = self.build_request(input)
        response = grpc_stub.Infer(request)

        print(response.request_status.msg)
        print(response)
        results = {}
        for meta, raw_output in zip(self.model.output, response.raw_output):
            data_type = DTYPE[meta.data_type]
            print(raw_output)
            np_data = np.frombuffer(raw_output, dtype=data_type)
            results[meta.name] = np_data

        return results
