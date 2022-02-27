import os
import subprocess

import numpy as np
from loguru import logger


def onnx2trt(onnx_path, engine_path=None, dtype='fp32', trtexec_path='trtexec', workspace_size=2048, use_sparsity=True):
    """ Optimize onnx model to tensorrt using trtexec.

    Args:
        onnx_path (str): Path to onnx model.
        engine_path (str): Path to save engine.
        dtype (str or np.dtype): Model data type. (fp32, fp16 or int8).
        trtexec_path (str): Path to trtexec binary.
        workspace_size (int): Size of workspace in MB.
        use_sparsity (bool): Use sparsity boolean (check NVidia docs for more info).
    """

    # Check dtype
    valid_dtypes = {
        np.int8: 'int8',
        np.float16: 'fp16',
        np.float32: 'fp32',
        'noTF32': 'noTF32'
    }
    dtype = dtype.lower()
    if dtype not in valid_dtypes.values():
        if dtype in valid_dtypes.keys():
            dtype = valid_dtypes[dtype]
        else:
            raise ValueError('Invalid dtype: "%s". Should be one of: %s' % (dtype, ', '.join(valid_dtypes.values())))

    # Check paths
    if not os.path.isfile(onnx_path):
        raise ValueError('Path to onnx model "%s" does not exist' % onnx_path)

    if not engine_path:
        engine_path = "".join(onnx_path.split(".")[:-1]) + "_" + dtype + ".engine"

    # Run optimizer
    script = [trtexec_path]  # Execute trtexec binary
    script.append("--onnx=" + onnx_path)  # Add path to onnx
    script.append("--saveEngine=" + engine_path)  # Add path to save engine
    script.append("--workspace=" + str(workspace_size))  # Add workspace size
    if dtype != 'fp32':
        script.append("--" + dtype)  # Add dtype
    if use_sparsity:
        script.append("--sparsity=enable")
    subprocess.check_call(script)

    logger.info("Successfully converted model from ONNX (%s) to TensorRT (%s)" % (onnx_path, engine_path))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Optimize model from ONNX to TensorRT engine')
    parser.add_argument('onnx_path', help='Path to onnx model file.', type=str)
    parser.add_argument('--engine_path', help='Path to save TensorRT engine.', type=str, default=None)
    parser.add_argument('--dtype', help='Model data type. (fp32, fp16 or int8)', type=str, default='fp32')
    parser.add_argument('--trtexec_path', help='Path to trtexec binary.', type=str,
                        default='/usr/local/tensorrt-8.2.2/targets/x86_64-linux-gnu/bin/trtexec')
    parser.add_argument('--workspace_size', help='Set vRAM workspace size for optimizer', type=int, default=1024)

    args = parser.parse_args()
    onnx2trt(args.onnx_path, args.engine_path, args.dtype, args.trtexec_path, args.workspace_size)
