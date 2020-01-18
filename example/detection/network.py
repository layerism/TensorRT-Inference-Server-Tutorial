import argparse
import os
import time
from builtins import range
from functools import partial

import numpy as np


if __name__ == "__main__":
    from trtis import onnx_backend
    from trtis import trt_backend

    from network import dla34, resnet
    model = dla34.get_pose_net(34, {'hm': 1, 'wh': 2, 'reg': 2}, 256)
    #model = resnet.get_pose_net(18, {'hm': 1, 'wh': 2, 'reg': 2}, 64)

    inputs_def = [
        {
            "name": "process_img",
            "dims": [1, 3, 512, 512],
            "data_type": "TYPE_FP32"
        }
    ]

    outputs_def = [
        {
            "name": "heatmap",
            "dims": [1, 1, 128, 128],
            "data_type": "TYPE_FP32"
        },
        {
            "name": "bbox_wh",
            "dims": [1, 2, 128, 128],
            "data_type": "TYPE_FP32"
        },
        {
            "name": "center_shift",
            "dims": [1, 2, 128, 128],
            "data_type": "TYPE_FP32"
        }
    ]

    from pre_process import preprocess
    import tensorflow as tf
    sess = tf.Session()
    preprocess_fn = lambda x: sess.run(preprocess([x]))[0]

    stream = trt_backend.ImageBatchStream("./calibrator_files", 5, preprocess_fn)
    int8_calibrator = trt_backend.IInt8EntropyCalibrator2(inputs_def, stream)

    trt_backend.torch2trt(
        computation_graph=model,
        graph_name="detection-network",
        model_file="./network/dla34.pth",
        inputs_def=inputs_def,
        outputs_def=outputs_def,
        instances=16,
        gpus=[0, 1, 2, 3],
        version=1,
        export_path="../../model_repository",
        int8_calibrator=int8_calibrator
    )

    # onnx_backend.torch2onnx(
    #     computation_graph=model,
    #     graph_name="face-det-network",
    #     model_file="./network/dla34.pth",
    #     inputs_def=INPUT_DEF,
    #     outputs_def=OUTPUT_DEF,
    #     instances=16,
    #     gpus=[2],
    #     version=1,
    #     export_path="../../model_repository"
    # )
