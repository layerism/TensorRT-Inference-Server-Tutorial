import os

import numpy as np
import tensorflow as tf
# from tensorflow.saved_model import predict_signature_def, tag_constants

from .set_config import DTYPE, generate_trtis_config


def tf2savedmodel(
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
    os.system("rm -rf {}".format(export_path))
    #os.system("mkdir -p {}".format(export_path))
    #os.system("mkdir -p {}/{}".format(export_path, version))

    graph = tf.Graph()
    with graph.as_default() as g:
        dummy_inputs = []
        input_names = []
        for i, input in enumerate(inputs_def):
            name = input.get("name", None)
            shape = input.get("dims", None)
            dtype = DTYPE[input.get("data_type", None)]
            if dtype == tf.string:
                dummy_input = tf.placeholder(dtype, [], name=name)
            else:
                dummy_input = tf.placeholder(dtype, shape, name=name)
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
            dtype = DTYPE[output.get("data_type", None)]
            output = tf.identity(dummy_outputs[i], name=name)
            output_names.append(name)

    model_path = os.path.join(export_path, str(version))
    with tf.Session(graph=graph) as sess:
        tf.saved_model.simple_save(
            sess, model_path,
            inputs=dict(zip(input_names, dummy_inputs)),
            outputs=dict(zip(output_names, dummy_outputs))
        )

    os.system("mv {}/saved_model.pb {}/model.savedmodel".format(model_path, model_path))

    if gen_trtis_config:
        generate_trtis_config(
            graph_name=graph_name,
            inputs_def=inputs_def,
            outputs_def=outputs_def,
            max_batch_size=max_batch_size,
            instances=instances,
            gpus=gpus,
            export_path=export_path,
            verbose=verbose
        )
