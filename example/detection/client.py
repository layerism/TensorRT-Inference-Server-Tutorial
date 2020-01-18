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

#import trt_client
from trt_client import client, client_grpc


class ManagerWatchdog(object):

    def __init__(self):
        self.manager_pid = os.getppid()
        self.manager_dead = False

    def is_alive(self):
        if not self.manager_dead:
            self.manager_dead = os.getppid() != self.manager_pid
        return not self.manager_dead


class TestMultipleRun(object):

    def __init__(self, datas, FLAGS, n_worker=12):
        self.FLAGS = FLAGS
        self.datas = datas
        self.workers = []
        self.n_worker = n_worker
        self.result_queue = mp.Queue()
        self._lock = threading.Lock()

        self.worker_queue_idx_cycle = itertools.cycle(range(n_worker))

        self.runner = client.Inference(
            url=self.FLAGS.url,
            model_name=self.FLAGS.model_name,
            model_version=self.FLAGS.model_version
        )

    def inference(self, result_queue, data_queue):
        # runner = trt_client.Inference(
        #     url=self.FLAGS.url,
        #     model_name=self.FLAGS.model_name,
        #     model_version=self.FLAGS.model_version,
        #     protocol=self.FLAGS.protocol
        # )
        watch_dog = ManagerWatchdog()
        #sub_result_queue = mp.Queue()
        while watch_dog.is_alive():
            idx, data = data_queue.get(block=True)
            #self.runner.async_run(input=data, input_id=idx, result_queue=sub_result_queue)
            results = self.runner.run(input=data)
            #print("=========")
            result_queue.put(results)

    def start(self):
        self.data_queues = []
        for i in range(self.n_worker):
            data_queue = mp.Queue()
            work = mp.Process(target=self.inference, args=(self.result_queue, data_queue))
            self.workers.append(work)
            self.data_queues.append(data_queue)

        for work in self.workers:
            work.daemon = True
            work.start()

    def try_put_index(self):
        try:
            data = next(self.iterable_datas)
        except:
            return True

        for _ in range(self.n_worker):
            worker_queue_idx = next(self.worker_queue_idx_cycle)
            if self.workers[worker_queue_idx].is_alive():
                break

        self.data_queues[worker_queue_idx].put(data)

        return False

    def __iter__(self):
        self.iterable_datas = iter(self.datas)
        self.start()
        self._count = 0
        for i in range(10):
            self.try_put_index()
        return self

    def __next__(self):
        if self._count == len(self.datas):
            self.terminate_all()
            raise StopIteration

        self.try_put_index()
        results = self.result_queue.get(block=True)
        #input_id, results = self.runner.get(self.result_queue, block=True)
        self._count += 1

        return results

    def __len__(self):
        return len(self.datas)

    def terminate_all(self):
        with self._lock:
            for p in self.workers:
                if p.is_alive():
                    p.terminate()
        self.workers = []



def test_async_speed(runner, data, N=128, loop=10):
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
        print((time.time() - t0) / N)


def test_speed(runner, data, N=128, loop=10):
    for i in range(loop):
        t0 = time.time()
        for i in range(N):
            results = runner.run(input={"raw_image": data})
            #print("Time: {}-{}".format(image_id, stat))
            #for key, value in results.items():
            #    print(value[0].shape, value[0].dtype, key)
        print("=====================")
        print((time.time() - t0) / N)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str)
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--model-version', type=int, default=1)
    parser.add_argument('--protocol', type=str, default='gRPC')
    parser.add_argument('--image', type=str)
    FLAGS = parser.parse_args()

    runner = client.Inference(
        url=FLAGS.url,
        model_name=FLAGS.model_name,
        model_version=FLAGS.model_version
    )

    #image_np = trt_client.load_image(FLAGS.image, size=512, format="NCHW")
    #image_np = (np.random.random([1, 3, 512, 512]) * 255).astype(np.uint8)
    image_np = np.array([open(FLAGS.image, "rb").read()], dtype=bytes)
    #image_np = np.array(["1244556"], dtype=np.object)
    #image_np = open(FLAGS.image, 'rb').read()

    # image_np = cv2.imread(FLAGS.image)
    # image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    # image_np = np.array([image_np]).tobytes()
    # print(image_np.shape)

    test_speed(runner, image_np)
    #test_async_speed(runner, image_np)

    # results = runner.run(input={"raw_image": image_np})
    # img = cv2.imread(FLAGS.image)
    # for score, bbox in zip(results["score"][0][0], results["bbox"][0][0]):
    #     if score >= 0.3:
    #         cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    # cv2.imwrite('./xxx.jpg', img)

    # N = 10240
    # datas = list(zip(range(N), [{"raw_image": image_np}] * N))
    # runner = TestMultipleRun(datas, FLAGS, 2)
    #
    # for results in tqdm(runner):
    #     pass
