import os

import torch
import torch.nn as nn
from torch.nn import functional

from torchvision import transforms

TOP_K = 100
HEATMAP_SIZE = 128
N_CATEGORY = 1


def pixel_nms(heatmap, ksize=3, cutoff=1e-7):
    pool_heatmap = functional.max_pool2d(
        heatmap, kernel_size=ksize, stride=1, padding=ksize // 2
    )
    zeros = torch.zeros_like(heatmap)
    score = torch.sigmoid(heatmap)
    pool_score = torch.sigmoid(pool_heatmap)
    diff = torch.abs(score - pool_score)
    score = torch.where(diff < cutoff, score, zeros)
    return score


def get_coord(score, topk=TOP_K):
    # score.shape = [N, C, 128, 128]
    # score, category, indices = [N, topk]
    score = score.flatten(2, 3)  # [N, C, 128 * 128]
    score, category = score.max(1)  # [N, 128 * 128]
    score, indices = score.topk(topk, dim=1)
    indices = indices.flatten()
    category = category.flatten().index_select(0, indices).to(torch.int32)
    return score, category, indices


def get_bbox_wh(bbox_wh, indices):
    # bbox_wh.shape = [N, 2, 128, 128]
    bbox_wh = bbox_wh.flatten(2, 3)
    bbox_w, bbox_h = bbox_wh[:, 0, :], bbox_wh[:, 1, :]
    bbox_w = bbox_w.flatten().index_select(0, indices)
    bbox_h = bbox_h.flatten().index_select(0, indices)
    return (bbox_w, bbox_h)


def get_center_shift(center_shift, indices):
    bsize = center_shift.shape[0]
    center_shift = center_shift.flatten(2, 3)
    shift_w, shift_h = center_shift[:, 0, :], center_shift[:, 1, :]
    shift_w = shift_w.flatten().index_select(0, indices)
    shift_h = shift_h.flatten().index_select(0, indices)
    return (shift_w, shift_h)


def remap_bbox_to_raw(trans_mat, x, y):
    # trans_mat.shape = [2, 3]
    # x.shape = [100]
    # y.shape = [100]
    x_ = x * trans_mat[0, 0] + y * trans_mat[0, 1] + trans_mat[0, 2]
    y_ = x * trans_mat[1, 0] + y * trans_mat[1, 1] + trans_mat[1, 2]
    return (x_, y_)


def cvt_to_coord(indices, bbox_wh, shift_wh, trans_mat, shape):
    # indices, bbox_wh, shift_wh = [N, topk]
    # indices = indices.to(torch.float32)
    cy_coord = (indices // shape[2]).to(torch.float32)
    cx_coord = (indices - cy_coord * shape[2]).to(torch.float32)
    cx_coord += shift_wh[0]
    cy_coord += shift_wh[1]

    x0 = cx_coord - bbox_wh[0] / 2
    y0 = cy_coord - bbox_wh[1] / 2
    x1 = cx_coord + bbox_wh[0] / 2
    y1 = cy_coord + bbox_wh[1] / 2

    x0, y0 = remap_bbox_to_raw(trans_mat, x0, y0)
    x1, y1 = remap_bbox_to_raw(trans_mat, x1, y1)

    bbox = torch.stack([x0, y0, x1, y1], 1)

    return bbox


def postprocess(inputs):
    heatmap, bbox_wh, center_shift, trans_mat = inputs
    score = pixel_nms(heatmap)
    score, category, indices = get_coord(score)
    bbox_wh = get_bbox_wh(bbox_wh, indices)
    shift_wh = get_center_shift(center_shift, indices)

    shape = torch.tensor(heatmap.shape, dtype=torch.long)
    bbox = cvt_to_coord(indices, bbox_wh, shift_wh, trans_mat, shape)

    score = score.reshape(1, TOP_K)
    category = category.reshape(1, TOP_K)
    bbox = bbox.reshape(1, TOP_K, 4)

    return score, category, bbox


if __name__ == "__main__":
    from trtis import onnx_backend

    inputs_def = [
        {
            "name": "heatmap",
            "dims": [1, N_CATEGORY, HEATMAP_SIZE, HEATMAP_SIZE],
            "data_type": "TYPE_FP32"
        },
        {
            "name": "bbox_wh",
            "dims": [1, 2, HEATMAP_SIZE, HEATMAP_SIZE],
            "data_type": "TYPE_FP32"
        },
        {
            "name": "center_shift",
            "dims": [1, 2, HEATMAP_SIZE, HEATMAP_SIZE],
            "data_type": "TYPE_FP32"
        },
        {
            "name": "affine_trans_mat",
            "dims": [2, 3],
            "data_type": "TYPE_FP32"
        }
    ]

    outputs_def = [
        {
            "name": "score",
            "dims": [1, TOP_K],
            "data_type": "TYPE_FP32",
        },
        {
            "name": "category",
            "dims": [1, TOP_K],
            "data_type": "TYPE_INT32",
        },
        {
            "name": "bbox",
            "dims": [1, TOP_K, 4],
            "data_type": "TYPE_FP32",
        }
    ]

    onnx_backend.torch2onnx(
        computation_graph=postprocess,
        graph_name="detection-postprocess",
        model_file=None,
        inputs_def=inputs_def,
        outputs_def=outputs_def,
        instances=16,
        gpus=[0, 1, 2, 3],
        version=1,
        export_path="../../model_repository"
    )
