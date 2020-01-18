#!python
import argparse
import itertools
import multiprocessing as mp
import os
import sys
import threading
import time
from builtins import range
from functools import partial

import cv2
import numpy as np
from tqdm import tqdm

from trt_client import client


def test_async_speed(runner, data, N=2048, loop=10):
    for i in range(loop):
        t0 = time.time()
        for i in range(N):
            runner.async_run(
                input={"raw_image": data},
                input_id="det_{}".format(i),
            )

        for i in range(N):
            input_id, results = runner.get_result(block=True)

        print("=====================")
        print("QPS:          {:5d}".format(int(1.0 / ((time.time() - t0) / N))))
        print("TIME/PER-IMG: {:5.3f}(ms)".format((time.time() - t0) / N * 1000.0))


def test_speed(runner, data, N=512, loop=10):
    for i in range(loop):
        t0 = time.time()
        for i in range(N):
            results = runner.run(input={"raw_image": data})
        print("=====================")
        print("QPS:          {:5d}".format(int(1.0 / ((time.time() - t0) / N))))
        print("TIME/PER-IMG: {:5.3f}(ms)".format((time.time() - t0) / N * 1000.0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str)
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--model-version', type=int, default=1)
    parser.add_argument('--image', type=str)
    FLAGS = parser.parse_args()

    runner = client.Inference(
        url=FLAGS.url,
        model_name=FLAGS.model_name,
        model_version=FLAGS.model_version
    )

    image_np = np.array([open(FLAGS.image, "rb").read()], dtype=bytes)

    #test_speed(runner, image_np)
    test_async_speed(runner, image_np)

    results = runner.run(input={"raw_image": image_np})
    img = cv2.imread(FLAGS.image)
    for score, bbox in zip(results["score"][0][0], results["bbox"][0][0]):
        if score >= 0.4:
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    cv2.imwrite('./xxx.jpg', img)
