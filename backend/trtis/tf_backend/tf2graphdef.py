import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util

from trtis.set_config import TF_DTYPE, generate_trtis_config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def tf2graphdef(
    computation_graph,
    graph_name="model",
    model_file=None,
    inputs_def=[{
        "name": None, "dims": []
    }],
    outputs_def=[{
        "name": None, "dims": []
    }],
    instances=1,
    gpus=[0],
    version=1,
    export_path="./",
    max_batch_size=0,
    device="cuda",
    gen_trtis_config=True,
    verbose=True
):

    export_path = os.path.join(export_path, graph_name)
    model_path = os.path.join(export_path, str(version), "model.graphdef")
    os.system("rm -rf {}".format(export_path))
    os.system("mkdir -p {}".format(export_path))
    os.system("mkdir -p {}/{}".format(export_path, version))

    graph = tf.Graph()
    with graph.as_default() as g:
        dummy_inputs = []
        input_names = []
        for i, input in enumerate(inputs_def):
            name = input.get("name", None)
            shape = input.get("dims", None)
            dtype = TF_DTYPE[input.get("data_type", None)]
            tf_shape = list(map(lambda dim: None if dim is -1 else dim, shape))
            dummy_input = tf.placeholder(dtype, tf_shape, name=name)
            #dummy_input = (torch.rand(shape) * 255).to(torch.uint8)
            #dummy_inputs.append({name: dummy_input})
            dummy_inputs.append(dummy_input)
            input_names.append(name)

        dummy_outputs = computation_graph(*dummy_inputs)
        if type(dummy_outputs) not in [list, tuple]:
            dummy_outputs = [dummy_outputs]

        output_names = []
        for i, output in enumerate(outputs_def):
            name = output.get("name", None)
            shape = output.get("dims", None)
            dtype = TF_DTYPE[output.get("data_type", None)]
            output = tf.identity(dummy_outputs[i], name=name)
            output_names.append(name)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = ",".join(map(str, gpus))
    with tf.Session(graph=graph, config=config) as sess:
        frozen_graph_def = graph_util.convert_variables_to_constants(
            sess, sess.graph_def, output_names
        )

        with open(model_path, 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())

    if gen_trtis_config:
        generate_trtis_config(
            graph_name=graph_name,
            platform="tensorflow_graphdef",
            inputs_def=inputs_def,
            outputs_def=outputs_def,
            max_batch_size=max_batch_size,
            instances=instances,
            gpus=gpus,
            export_path=export_path,
            verbose=verbose
        )
