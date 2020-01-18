#!/python
import argparse
import json
import os
from functools import partial

import numpy as np
import torch
import torch.nn as nn

import onnx as onnx
from onnx import shape_inference, optimizer

from trtis.set_config import TORCH_DTYPE, generate_trtis_config
from trtis.onnx_backend import onnxsim


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


def modify_to_dynamic_shape(model, input_names, output_names):
    for input in model.graph.input:
        if input.name not in input_names:
            continue
        input.type.tensor_type.shape.dim[0].dim_param = '?'
        #input.type.tensor_type.shape.dim[1].dim_param = '?'
        #input.type.tensor_type.shape.dim[2].dim_param = '?'
        #input.type.tensor_type.shape.dim[3].dim_param = '?'
    for output in model.graph.output:
        if output.name not in output_names:
            continue
        #for i in range(len(output.type.tensor_type.shape.dim)):
        #    output.type.tensor_type.shape.dim[i].dim_param = '?'

    return model


def optim_onnx(onnx_path, verbose=True):
    model = onnx.load(onnx_path)
    print("Begin Simplify ONNX Model ...")
    passes = [
        "eliminate_deadend",
        "eliminate_identity",
        "extract_constant_to_initializer",
        "eliminate_unused_initializer",
        "fuse_add_bias_into_conv",
        "fuse_bn_into_conv",
        "fuse_matmul_add_bias_into_gemm"
    ]
    model = optimizer.optimize(model, passes)
    #model = shape_inference.infer_shapes(model)
    #model = onnxsim.simplify(model)

    if verbose:
        for m in onnx.helper.printable_graph(model.graph).split("\n"):
            print(m)

    return model


def torch2onnx(
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
    export_path="./",
    opset_version=10,
    max_batch_size=0,
    device="cuda",
    gen_trtis_config=True,
    verbose=True
):

    if not isinstance(computation_graph, nn.Module):
        model = WrapperFunc(computation_graph)
    else:
        model = WrapperModel(computation_graph)

    if device == "cuda":
        model = model.cuda()
    else:
        model = model.cpu()

    # model key    : A.B.x.x.x.x
    # pth-file key : C.A.E.x.x.x.x
    # load the model by non-exactly match pth-file
    if model_file is not None:
        checkpoint = torch.load(model_file)['state_dict']
        print("loading pt-file finishing .... ")
        state_dict = {}
        required_keys = list(model.state_dict().keys())
        # for required_key in required_keys:
        #     value = checkpoint.get(required_key.split(".", 1)[1], None)
        #     if value is None:
        #         print("missing key: {}".format(required_key))
        #         continue
        #     state_dict[required_key] = value

        for required_key, (key, value) in zip(required_keys, checkpoint.items()):
            if required_key.endswith(key.split(".", 1)[1]):
                print("pth-key: [{:60s}] ---> model-key: [{}]".format(key, required_key))
            else:
                print("pth-key: [{:60s}] -\\-> model-key: [{}]".format(key, required_key))
                continue

            state_dict[required_key] = value
        model.load_state_dict(state_dict)
        print("loading model finishing .... ")

    dummy_inputs = []
    input_names = []
    for i, input in enumerate(inputs_def):
        name = input.get("name", None)
        shape = input.get("dims", None)
        dtype = TORCH_DTYPE[input.get("data_type", None)]
        dummy_input = torch.ones(shape).to(dtype)
        #dummy_input = (torch.rand(shape) * 255).to(torch.uint8)
        if device == "cuda":
            dummy_input = dummy_input.cuda()
        dummy_inputs.append(dummy_input)
        input_names.append(name)

    dummy_inputs = dummy_inputs[0] if len(dummy_inputs) == 1 else dummy_inputs

    output_names = []
    for i, output in enumerate(outputs_def):
        shape = output.get("dims", None)
        name = output.get("name", None)
        output_names.append(name)

    export_path = os.path.join(export_path, graph_name)
    os.system("mkdir -p {}".format(export_path))
    os.system("mkdir -p {}/{}".format(export_path, version))

    onnx_path = "{}/{}/{}.onnx".format(export_path, version, "model")

    torch.onnx.export(
        model,
        dummy_inputs,
        onnx_path,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        keep_initializers_as_inputs=True,
        #dynamic_axes=dict(dynamic_batches)
    )

    model = optim_onnx(onnx_path)
    onnx.save(model, onnx_path)
    os.system("python -m onnxsim {} {}".format(onnx_path, onnx_path))

    if gen_trtis_config:
        generate_trtis_config(
            graph_name=graph_name,
            platform="onnxruntime_onnx",
            inputs_def=inputs_def,
            outputs_def=outputs_def,
            max_batch_size=max_batch_size,
            instances=instances,
            gpus=gpus,
            export_path=export_path,
            verbose=verbose
        )

    return model, onnx_path
