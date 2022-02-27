import numpy as np
import onnx
from onnx_tensorrt import backend

if __name__ == '__main__':
    model = onnx.load("/home/rizhik/optimize_tensorrt/weights/mobilenet_sim.onnx")
    engine = backend.prepare(model, device='CUDA:0', dtype=np.float32)
