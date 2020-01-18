import argparse
import multiprocessing as mp
import os
import time
from builtins import range
from functools import partial

import tensorrtserver.api.model_config_pb2 as model_config
import tensorrtserver.cuda_shared_memory as cudashm
from tensorrtserver.api import *

if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue

DTYPE = {2: np.uint8, 11: np.float32, 8: np.int32}


def create_cuda_shm(data, name, url, protocol, is_input=True):
    #c, h, w = shape
    shared_memory_ctx = SharedMemoryControlContext(url, protocol)
    byte_size = data.size * data.itemsize
    shm_handle = cudashm.create_shared_memory_region(name, byte_size, 3)

    if is_input:
        cudashm.set_shared_memory_region(shm_handle, [data])
        shared_memory_ctx.cuda_register(shm_handle)
    else:
        shared_memory_ctx.cuda_register(shm_handle)


def get_server_status(url, protocol, model_name, verbose=False):
    protocol = ProtocolType.from_str(protocol)
    ctx = ServerStatusContext(url, protocol, model_name, verbose)
    server_status = ctx.get_server_status()

    return server_status


def parse_model(url, protocol, model_name, verbose=False):
    protocol = ProtocolType.from_str(protocol)
    ctx = ServerStatusContext(url, protocol, model_name, verbose)
    server_status = ctx.get_server_status()

    if model_name not in server_status.model_status:
        raise Exception("unable to get status for '" + model_name + "'")

    status = server_status.model_status[model_name]
    config = status.config

    return config


class Inference(object):

    def __init__(self, url, model_name, model_version, protocol='gRPC'):
        model = parse_model(url, protocol, model_name)
        protocol = ProtocolType.from_str(protocol)
        self.model = model
        self.ctx = InferContext(
            url=url,
            protocol=protocol,
            model_name=model_name,
            model_version=model_version,
            verbose=False,
            streaming=False
        )
        self.result_queue = mp.Queue()

        self.outputs = {}
        for output in self.model.output:
            self.outputs[output.name] = InferContext.ResultFormat.RAW

    def callback(self, input_id, result_queue, infer_ctx, request_id):
        result_queue.put((request_id, input_id))

    def async_run(self, input, input_id):
        for key in input.keys():
            if type(input[key]) not in [list, tuple]:
                input[key] = [input[key]]

        callback_fn = partial(self.callback, input_id, self.result_queue)
        self.ctx.async_run(callback_fn, input, self.outputs)

    def run(self, input):
        for key in input.keys():
            if type(input[key]) not in [list, tuple]:
                input[key] = [input[key]]

        results = self.ctx.run(input, self.outputs, batch_size=1)
        return results

    def get_time(self):
        stat = self.ctx.get_stat()

        count = stat["completed_request_count"]
        request_dt = stat["cumulative_total_request_time_ns"] / 1.0e6
        send_dt = stat["cumulative_send_time_ns"] / 1.0e6
        receive_dt = stat["cumulative_receive_time_ns"] / 1.0e6
        inference_dt = (request_dt - send_dt - receive_dt)
        stat = {
            "completed_request_count": stat["completed_request_count"],
            "send_time_ms": float("{:5.3f}".format(send_dt / count)),
            "inference_time_ms": float("{:5.3f}".format(inference_dt / count)),
            "receive_time_ms": float("{:5.3f}".format(receive_dt / count))
        }
        return stat

    def get_result(self, block=True):
        (request_id, input_id) = self.result_queue.get(block=block)
        results = self.ctx.get_async_run_results(request_id)

        return input_id, results


class ManagerWatchdog(object):

    def __init__(self):
        self.manager_pid = os.getppid()
        self.manager_dead = False

    def is_alive(self):
        if not self.manager_dead:
            self.manager_dead = os.getppid() != self.manager_pid
        return not self.manager_dead


# class Parallel(object):
#
#     def __init__(self, worker=8):
#         self.inference = Inference()
#
#     def run(self, )
#     def start(self, )
