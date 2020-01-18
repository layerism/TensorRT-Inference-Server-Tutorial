import cv2
import numpy as np



def affine_transform(point_2d, affine_mat):
    x, y = point_2d
    x_ = affine_mat[0, 0] * x + affine_mat[0, 1] * y + affine_mat[0, 2]
    y_ = affine_mat[1, 0] * x + affine_mat[1, 1] * y + affine_mat[1, 2]
    return (x_, y_)


def get_affine_transform(src_shape, dst_shape, inv=False):
    src_shift = max(src_shape) / 2.0
    src_point1 = (src_shape[0] / 2.0, src_shape[1] / 2.0)
    src_point2 = (src_shape[0] / 2.0, src_shape[1] / 2.0 - src_shift)
    src_point3 = (src_shape[0] / 2.0 - src_shift, src_shape[1] / 2.0)

    dst_shift = max(dst_shape) / 2.0
    dst_point1 = (dst_shape[0] / 2.0, dst_shape[1] / 2.0)
    dst_point2 = (dst_shape[0] / 2.0, dst_shape[1] / 2.0 - dst_shift)
    dst_point3 = (dst_shape[0] / 2.0 - dst_shift, dst_shape[1] / 2.0)

    src_points = np.float32([src_point1, src_point2, src_point3])
    dst_points = np.float32([dst_point1, dst_point2, dst_point3])

    if inv is False:
        affine_mat = cv2.getAffineTransform(src_points, dst_points)
    else:
        affine_mat = cv2.getAffineTransform(dst_points, src_points)

    return affine_mat


def resize_and_pad(image, size=(512, 512)):
    size_h, size_w = size
    h, w, c = image.shape
    if h >= w:
        new_w = int((size_h / float(h)) * w)
        new_h = size_h
    elif h < w:
        new_h = int((size_w / float(w)) * h)
        new_w = size_w

    image = cv2.resize(image, (new_w, new_h))

    if size_h != new_h or size_w != new_w:
        pad_w_0 = int((size_w - new_w) / 2)
        pad_w_1 = size_w - new_w - pad_w_0
        pad_h_0 = int((size_h - new_h) / 2)
        pad_h_1 = size_h - new_h - pad_h_0
        image = cv2.copyMakeBorder(
            image, pad_h_0, pad_h_1, pad_w_0, pad_w_1, cv2.BORDER_CONSTANT
        )

    return image
