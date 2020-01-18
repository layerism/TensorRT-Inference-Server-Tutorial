#!/python
import argparse
import json
import os
import torch
import tensorflow as tf


TORCH_DTYPE = {
    "TYPE_FP32": torch.float32,
    "TYPE_INT32": torch.int32,
    "TYPE_UINT8": torch.uint8,
    "TYPE_FP16": torch.float16
}


TF_DTYPE = {
    "TYPE_FP32": tf.float32,
    "TYPE_INT32": tf.int32,
    "TYPE_UINT8": tf.uint8,
    "TYPE_FP16": tf.float16,
    "TYPE_STRING": tf.string
}


TENSOR = """\
  {{
    name: "{name}",
    dims: {dims},
    data_type: {data_type}
  }}\
"""


TENSOR_RESHAPE = """\
  {{
    name: "{name}",
    dims: {dims},
    data_type: {data_type},
    reshape: {{ shape: {reshape} }}
  }}\
"""


CONFIG_PBTXT = """\
name: "{name}"
platform: "{platform}"
version_policy: {{ all {{ }} }}
max_batch_size: {max_batch_size}
input {input}
output {output}
instance_group [
  {{
    count: {instances}
    kind: KIND_GPU
    gpus: {gpus}
  }}
]\
"""


def data_def_dumps(node_def, remove_batch_dim=False):
    data = []
    for node in node_def:
        name = node.get("name", None)
        dims = node.get("dims", None)
        data_type = node.get("data_type", "TYPE_FP32")
        #format = node.get("format", "FORMAT_NONE")
        reshape = node.get("reshape", dims)
        if remove_batch_dim is False:
            format_output = TENSOR.format(
                name=name,
                dims=dims,
                data_type=data_type
            )
        else:
            format_output = TENSOR_RESHAPE.format(
                name=name,
                dims=dims,
                data_type=data_type,
                reshape=dims[1:]
            )
        data.append(format_output)

    data = "[\n" + ",\n".join(data) + "\n]"
    return data


def generate_trtis_config(
    graph_name="dd",
    platform="tensorrt_plan",
    inputs_def={},
    outputs_def={},
    max_batch_size=1,
    instances=1,
    gpus=[0],
    export_path="./",
    verbose=True
):

    remove_batch_dim = True if platform is "tensorrt_plan" else False

    config_path = "{}/config.pbtxt".format(export_path)
    config_content = CONFIG_PBTXT.format(
        name=str(graph_name),
        platform=str(platform),
        max_batch_size=max_batch_size,
        input=data_def_dumps(inputs_def, remove_batch_dim),
        output=data_def_dumps(outputs_def, remove_batch_dim),
        instances=instances,
        gpus=gpus
    )

    with open(config_path, "w") as cfg:
        cfg.write(config_content)

    if verbose:
        print(config_content)
