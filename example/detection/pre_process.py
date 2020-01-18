import os

import numpy as np
import tensorflow as tf

HEATMAP_SIZE = 128
MEAN = (0.408, 0.447, 0.470)
STD = (0.274, 0.274, 0.274)


def backward_affine_transform(raw_shape, dst_shape=(HEATMAP_SIZE, HEATMAP_SIZE)):
    """
    shape:
        dst(3, 3) x T(3, 2) = src(3, 2)
    solve T matrix:
        |d11, d12, 1|   |T11, T12|   |s11, s12|
        |d21, d22, 1| x |T21, T22| = |s21, s22|
        |d31, d32, 1|   |T31, T32|   |s31, s32|
    output:
        T(3, 2).transpose = T(2, 3)
    """
    raw_shape = tf.cast(raw_shape, tf.float32)
    src_cy, src_cx = raw_shape[0] / 2.0, raw_shape[1] / 2.0
    src_dc = tf.math.maximum(src_cx, src_cy)
    src_points = tf.stack([
        [src_cx, src_cy],
        [src_cx, src_cy - src_dc],
        [src_cx - src_dc, src_cy]
    ])

    dst_shape = tf.cast(dst_shape, tf.float32)
    dst_cy, dst_cx = dst_shape[0] / 2.0, dst_shape[1] / 2.0
    dst_dc = tf.math.maximum(dst_cx, dst_cy)
    dst_points = tf.stack([
        [dst_cx, dst_cy, 1],
        [dst_cx, dst_cy - dst_dc, 1],
        [dst_cx - dst_dc, dst_cy, 1]
    ])

    trans_mat = tf.linalg.solve(dst_points, src_points, name="affine")
    trans_mat = tf.transpose(trans_mat, perm=(1, 0))

    return trans_mat


def image_decoder(image_bytes):
    with tf.name_scope("image_decoder") as scope:
        image = tf.io.decode_image(
            image_bytes[0],
            channels=None,
            dtype=tf.dtypes.uint8,
            name="decode_image",
            expand_animations=False
        )
        image = tf.expand_dims(image, 0) # [1, H, W, 3]
        raw_shape = tf.shape(image)[1:3]
        affine_trans_mat = backward_affine_transform(raw_shape)
        return image, affine_trans_mat


def _resize_and_pad(image, size=512):
    with tf.name_scope("image_resize") as scope:
        image = tf.image.resize_image_with_pad(
            image, target_height=size, target_width=size
        )
        image = tf.transpose(image, perm=[0, 3, 1, 2])
        return image


def _normalize(tensor, mean=MEAN, std=STD):
    with tf.name_scope("normalize") as scope:
        tensor = tf.cast(tensor, tf.float32)
        tensor = tensor / 255.0
        mean = tf.constant(mean, dtype=tf.float32, name="mean")
        mean = tf.reshape(mean, [1, 3, 1, 1])
        std = tf.constant(std, dtype=tf.float32, name="std")
        std = tf.reshape(std, [1, 3, 1, 1])
        tensor = (tensor - mean) / std
        return tensor


def preprocess(raw_image):
    image, affine_trans_mat = image_decoder(raw_image)
    image = _resize_and_pad(image)
    tensor = _normalize(image)
    return tensor, affine_trans_mat


if __name__ == "__main__":
    from trtis import tf_backend

    inputs_def = [
        {
            "name": "raw_image",
            "dims": [1],
            "data_type": "TYPE_STRING",
        }
    ]

    outputs_def = [
        {
            "name": "process_img",
            "dims": [1, 3, 512, 512],
            "data_type": "TYPE_FP32"
        },
        {
            "name": "affine_trans_mat",
            "dims": [2, 3],
            "data_type": "TYPE_FP32"
        }
    ]

    tf_backend.tf2graphdef(
        computation_graph=preprocess,
        graph_name="detection-preprocess",
        model_file=None,
        inputs_def=inputs_def,
        outputs_def=outputs_def,
        instances=16,
        gpus=[0, 1, 2, 3],
        version=1,
        export_path="../../model_repository"
    )
