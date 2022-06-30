import os

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
from loguru import logger

cuda.init()


class InferenceTRT:
    """Inference TensorRT engine."""

    def __init__(self, engine_path=None, device=None):
        """Initialize class variables.

        Args:
            engine_path (str): Path to TensorRT engine file.
            device (cuda.Device): Cuda device to run inference on.
        """

        # Initialize input variables
        if not os.path.isfile(engine_path):
            raise ValueError("Engine file not found: %s" % engine_path)
        self.engine_path = engine_path
        self.trt_logger = trt.Logger(trt.Logger.WARNING)

        # Initialize buffers variables
        self.h_input = None
        self.d_input = None
        self.h_output = None
        self.d_output = None

        # Initialize engine
        self.device = device if device is not None else cuda.Device(0)
        self.device_ctx = self.device.make_context()

        self.engine, self.engine_ctx = None, None
        if engine_path:
            self.load_engine(engine_path)

    def load_engine(self, engine_path):
        """Load TensorRT engine from file.

        Args:
            engine_path (str): Path to TensorRT engine file.
        """

        # Load engine
        with open(engine_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.engine_ctx = self.engine.create_execution_context()

        # Allocate buffers
        (
            self.h_input,
            self.d_input,
            self.h_output,
            self.d_output,
        ) = self._allocate_buffers()

        logger.info("Successfully loaded TensorRT engine from %s" % engine_path)

    def predict(self, inputs):
        """Get model prediction.

        Args:
            inputs (np.ndarray): Model inputs array.
        Returns:
            Model prediction as np.ndarray.
        """

        if self.device_ctx:
            self.device_ctx.push()

        np.copyto(self.h_input, inputs)
        self._do_inference()
        if self.device_ctx:
            self.device_ctx.pop()

        return self.h_output

    def _do_inference(self):
        stream = cuda.Stream()
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(self.d_input, self.h_input, stream)
        # Run inference.
        self.engine_ctx.execute_async_v2(
            bindings=[int(self.d_input), int(self.d_output)],
            stream_handle=stream.handle,
        )
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        # Return the host output.
        return self.h_output

    def _allocate_buffers(self):
        # Allocate input buffers.
        input_shape_dim = self.engine.get_binding_shape(0)
        # convert from tensorrt Dim to tuple
        input_shape = tuple([input_shape_dim[i] for i in range(len(input_shape_dim))])
        host_input = cuda.pagelocked_empty(input_shape, dtype=np.float32)

        # Allocate output buffers.
        output1_shape_dim = self.engine.get_binding_shape(1)
        # convert from tensorrt Dim to tuple
        output1_shape = tuple(
            [output1_shape_dim[i] for i in range(len(output1_shape_dim))]
        )
        host_output = cuda.pagelocked_empty(output1_shape, dtype=np.float32)

        device_input = cuda.mem_alloc(host_input.nbytes)
        logger.info(f"Allocated memory for input with shape {input_shape}")
        device_output = cuda.mem_alloc(host_output.nbytes)
        logger.info(f"Allocated memory for output with shape {output1_shape}")

        return host_input, device_input, host_output, device_output

    def __del__(self):
        if self.device_ctx:
            self.device_ctx.pop()
            logger.info("Released cuda context")
