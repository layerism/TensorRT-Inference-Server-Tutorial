import ctypes
import glob
import os
import random

import numpy as np

import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create_calibration_dataset(datasets_path, n=100):
    # Create list of calibration images (filename)
    # This sample code picks 100 images at random from training set
    datasets_path = os.path.join(datasets_path, "*")
    calibration_files = glob.glob(datasets_path)
    random.shuffle(calibration_files)
    return calibration_files[:n]


class ImageBatchStream(object):

    def __init__(self, calibration_path, batch_size, preprocessor):
        self.batch_size = batch_size
        self.preprocessor = preprocessor

        self.batch = 0
        calibration_files = create_calibration_dataset(calibration_path)
        self.max_batches = len(calibration_files) // batch_size
        self.max_batches += 1 if (len(calibration_files) % batch_size) else 0
        self.files = calibration_files

    def reset(self):
        self.batch = 0

    def next_batch(self, verbose=True):
        if self.batch < self.max_batches:
            start = self.batch_size * self.batch
            end = self.batch_size * (self.batch + 1)

            calibration_data = []
            for image_path in self.files[start:end]:
                if verbose:
                    print("[ImageBatchStream] Processing ", image_path)
                raw_image = open(image_path, "rb").read()
                img = self.preprocessor(raw_image)
                calibration_data.append(img)

            self.batch += 1
            return np.concatenate(calibration_data, 0)
        else:
            return np.array([])


class IInt8EntropyCalibrator(trt.IInt8EntropyCalibrator):

    def __init__(self, input_def, stream, cache_file="./calibration_cache.bin"):
        trt.IInt8EntropyCalibrator.__init__(self)

        self.input_layers = []
        for input in input_def:
            name = input.get("name", None)
            self.input_layers.append(name)

        self.stream = stream
        calibration_data = self.stream.next_batch(verbose=False)
        self.d_input = cuda.mem_alloc(calibration_data.nbytes)
        self.stream.reset()

        self.cache_file = cache_file
        os.system("rm -f {}".format(self.cache_file))

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self, names):
        data = self.stream.next_batch()
        cuda.memcpy_htod(self.d_input, data)

        if data.size == 0:
            return None
        else:
            return [int(self.d_input)]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again.
        # Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


class IInt8EntropyCalibrator2(trt.IInt8EntropyCalibrator2):

    def __init__(self, input_def, stream, cache_file="./calibration_cache.bin"):
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.input_layers = []
        for input in input_def:
            name = input.get("name", None)
            self.input_layers.append(name)

        self.stream = stream
        calibration_data = self.stream.next_batch(verbose=False)
        self.d_input = cuda.mem_alloc(calibration_data.nbytes)
        self.stream.reset()

        self.cache_file = cache_file
        os.system("rm -f {}".format(self.cache_file))

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self, names):
        data = self.stream.next_batch()
        cuda.memcpy_htod(self.d_input, data)

        if data.size == 0:
            return None
        else:
            return [int(self.d_input)]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again.
        # Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


class IInt8MinMaxCalibrator(trt.IInt8MinMaxCalibrator):

    def __init__(self, input_def, stream, cache_file="./calibration_cache.bin"):
        trt.IInt8MinMaxCalibrator.__init__(self)

        self.input_layers = []
        for input in input_def:
            name = input.get("name", None)
            self.input_layers.append(name)

        self.stream = stream
        calibration_data = self.stream.next_batch(verbose=False)
        self.d_input = cuda.mem_alloc(calibration_data.nbytes)
        self.stream.reset()

        self.cache_file = cache_file
        os.system("rm -f {}".format(self.cache_file))

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self, names):
        data = self.stream.next_batch()
        cuda.memcpy_htod(self.d_input, data)

        if data.size == 0:
            return None
        else:
            return [int(self.d_input)]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again.
        # Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


class IInt8LegacyCalibrator(trt.IInt8LegacyCalibrator):

    def __init__(self, input_def, stream, cache_file="./calibration_cache.bin"):
        trt.IInt8LegacyCalibrator.__init__(self)

        self.input_layers = []
        for input in input_def:
            name = input.get("name", None)
            self.input_layers.append(name)

        self.stream = stream
        calibration_data = self.stream.next_batch(verbose=False)
        self.d_input = cuda.mem_alloc(calibration_data.nbytes)
        self.stream.reset()

        self.cache_file = cache_file
        os.system("rm -f {}".format(self.cache_file))

        self.quantile = 0.0
        self.regression_cutoff = 0.0

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self, names):
        data = self.stream.next_batch()
        cuda.memcpy_htod(self.d_input, data)

        if data.size == 0:
            return None
        else:
            return [int(self.d_input)]

    def get_quantile(self):
        return self.quantile

    def get_regression_cutoff(self):
        return self.regression_cutoff

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again.
        # Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
