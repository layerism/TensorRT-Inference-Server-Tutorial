#!/python
import argparse
import json
import os
from functools import partial

import numpy as np
import torch
import torch.nn as nn

import onnx as onnx
import tensorrt as trt

from trtis.set_config import generate_trtis_config
from trtis import onnx_backend


class MergeModel(nn.Module):

    def __init__(self, model, preprocess, postprocess):
        super(MergeModel, self).__init__()
        self.model = model
        self.preprocess = preprocess
        self.postprocess = postprocess

    def forward(self, x):
        x = self.preprocess(x)
        x = self.model(x)
        x = self.postprocess(x)
        return x


class WrapperFunc(nn.Module):

    def __init__(self, func):
        super(WrapperFunc, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class WrapperModel(nn.Module):

    def __init__(self, model):
        super(WrapperModel, self).__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x


def GiB(val):
    return val * 1 << 30


def build_engine(onnx_path, export_path, int8_calibrator=None):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    trt_logger = trt.Logger()
    with trt.Builder(trt_logger) as builder:
        with builder.create_network() as network:
            with trt.OnnxParser(network, trt_logger) as parser:
                builder.max_workspace_size = GiB(1) # 1GB
                builder.max_batch_size = 1
                if int8_calibrator is not None:
                    builder.int8_mode = True
                    builder.int8_calibrator = int8_calibrator

                # Parse model file
                if not os.path.exists(onnx_path):
                    print('ONNX file {} not found'.format(onnx_path))
                    exit(0)

                print('Loading ONNX file from path {}...'.format(onnx_path))
                with open(onnx_path, 'rb') as model:
                    print('Beginning ONNX file parsing')
                    parser.parse(model.read())
                print('Completed parsing of ONNX file')

                print('Building an engine from file {} ...'.format(onnx_path))
                engine = builder.build_cuda_engine(network)

                print("Completed creating Engine")
                with open(export_path, "wb") as f:
                    f.write(engine.serialize())
                return engine


def torch2trt(
    computation_graph,
    graph_name="model",
    model_file=None,
    inputs_def=[{
        "name": None, "shape": []
    }],
    outputs_def=[{
        "name": None, "shape": []
    }],
    instances=1,
    gpus=[0],
    version=1,
    max_batch_size=1,
    int8_calibrator=None,
    export_path="./",
    onnx_opset_version=10,
    device="cuda",
    gen_trtis_config=True,
    verbose=True
):

    onnx_model, onnx_path = onnx_backend.torch2onnx(
        computation_graph=computation_graph,
        graph_name=graph_name,
        model_file=model_file,
        inputs_def=inputs_def,
        outputs_def=outputs_def,
        instances=instances,
        gpus=gpus,
        version=version,
        export_path=export_path,
        opset_version=onnx_opset_version,
        max_batch_size=max_batch_size,
        device=device,
        gen_trtis_config=False,
        verbose=verbose
    )

    export_path = os.path.join(export_path, graph_name)
    os.system("mkdir -p {}".format(export_path))
    os.system("mkdir -p {}/{}".format(export_path, version))
    trt_engine_path = "{}/{}/{}.plan".format(export_path, version, "model")
    build_engine(
        onnx_path=onnx_path,
        export_path=trt_engine_path,
        int8_calibrator=int8_calibrator
    )
    os.system("rm -rf {}".format(onnx_path))

    if gen_trtis_config:
        generate_trtis_config(
            graph_name=graph_name,
            platform="tensorrt_plan",
            inputs_def=inputs_def,
            outputs_def=outputs_def,
            max_batch_size=max_batch_size,
            instances=instances,
            gpus=gpus,
            export_path=export_path,
            verbose=verbose
        )
