import os

import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np
import tensorrt as trt


class ImageBatchStream:
    def __init__(self, shape, calibration_files, preprocessor):
        if len(shape) < 4:
            raise ValueError(f"Input shape must have at least 4 dimensions (with batch size), but {shape} given")

        self.shape = shape
        self.batch_size = shape[0]
        self.files = calibration_files
        self.preprocessor = preprocessor

        self.batch = 0
        self.max_batches = (len(calibration_files) // self.batch_size) + \
                           (1 if (len(calibration_files) % self.batch_size)
                            else 0)
        self.calibration_data = np.zeros(shape, dtype=np.float32)

    def reset(self):
        self.batch = 0

    def next_batch(self):
        if self.batch < self.max_batches:
            imgs = []
            files_for_batch = self.files[self.batch_size * self.batch:
                                         self.batch_size * (self.batch + 1)]
            for f in files_for_batch:
                # logger.info("[ImageBatchStream] Processing %s" % f)
                img = self.preprocessor(f)
                imgs.append(img)
            for i in range(len(imgs)):
                self.calibration_data[i] = imgs[i]
            self.batch += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])


class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_files, preprocessor, shape, cache_file="unknown_model.calib"):
        """ Int8 Calibrator.

        Args:
            calibration_files (list or tuple): List of paths to images for calibration.
            preprocessor (function): Image preprocessing function.
            shape (list or tuple): Input image shape. With batch_size.
            cache_file (str): Path to save calibration.
        """

        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file

        self.stream = ImageBatchStream(shape, calibration_files, preprocessor)
        self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self, names):
        batch = self.stream.next_batch()
        if not batch.size:
            return None
        cuda.memcpy_htod(self.d_input, batch)
        return [self.d_input]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
