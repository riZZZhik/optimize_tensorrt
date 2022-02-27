import os

import numpy as np
import tensorrt as trt
from loguru import logger


def onnx2trt_handmade(onnx_path, engine_path=None, dtype='fp32', calib=None):
    """ Convert model from ONNX to TensorRT engine.

    Args:
        onnx_path (str): Path to ONNX model.
        engine_path (str): Path to save TensorRT engine.
        dtype (str): Model data type. (fp32, fp16 or int8).
        calib (trt.IInt8EntropyCalibrator2): Calibration class. Only for INT8.
    """

    # Check input variables
    valid_dtypes = {
        np.int8: 'int8',
        np.float16: 'fp16',
        np.float32: 'fp32'
    }
    dtype = dtype.lower()
    if dtype not in valid_dtypes.values():
        if dtype in valid_dtypes.keys():
            dtype = valid_dtypes[dtype]
        else:
            raise ValueError('Invalid dtype: "%s". Should be one of: %s' % (dtype, ', '.join(valid_dtypes.values())))

    if not os.path.isfile(onnx_path):
        raise ValueError('Path to onnx model "%s" does not exist' % onnx_path)

    if not engine_path:
        engine_path = "".join(onnx_path.split(".")[:-1]) + "_" + dtype + ".engine"

    # Initialize builder and network
    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()

    # Parse onnx
    with trt.OnnxParser(network, trt_logger) as parser:
        parser.parse_from_file(onnx_path)

    # Build engine
    builder.max_batch_size = 1
    config.max_workspace_size = 2 << 30
    if dtype == 'fp16':
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("Using FP16 mode")
        else:
            logger.warning("Platform does not support FP16 mode. Using FP32 instead.")
    elif dtype == 'int8':
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calib
            logger.info("Using INT8 mode")
        else:
            logger.warning("Platform does not support INT8 mode. Using FP32 instead.")

    # Save engine to file
    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    logger.info("Successfully converted model from ONNX (%s) to TensorRT (%s)" % (onnx_path, engine_path))
