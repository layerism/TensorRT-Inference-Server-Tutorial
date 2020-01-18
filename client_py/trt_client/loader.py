import cv2
import numpy as np
from .transform import *


def load_image(image_files, size, format="NCHW", dtype=np.uint8):
    if type(image_files) not in [list, tuple]:
        image_files = [image_files]

    if type(size) not in [list, tuple]:
        size = (size, size)

    image_np = []
    for image_file in image_files:
        image = cv2.imread(image_file)
        image = resize_and_pad(image, size)
        image = np.array(image).astype(dtype)
        image = np.expand_dims(image, 0)
        if format == 'NCHW':
            image = np.transpose(image, (0, 3, 1, 2))
        image_np.append(image)

    return image_np
