import os

from loguru import logger


def fastai2onnx(
    fastai_path, input_shape, onnx_path=None, input_names=None, output_names=None
):
    """Convert FastAI model to ONNX.

    Args:
        fastai_path (str): Path to fastai model file.
        input_shape (tuple or list): Input shape (with batch dimension).
        onnx_path (str): Path to save converted model.
        input_names (tuple or list): Model's input names.
        output_names (tuple or list): Model's output names.
    """

    # Input requirements
    import torch
    from fastai.vision.all import load_learner, nn

    # Check input variables
    if not os.path.isfile(fastai_path):
        raise ValueError("FastAI model file not found in %s" % fastai_path)

    if onnx_path is None:
        onnx_path = "".join(fastai_path.split(".")[:-1]) + ".onnx"

    # Load FastAI model from file
    learn = load_learner(fastai_path)

    # Get PyTorch model
    pytorch_model = learn.model.eval()

    # Add Softmax layer to model (fastai thing)
    final_model = nn.Sequential(pytorch_model, torch.nn.Softmax(dim=1))

    # Export model to onnx
    torch.onnx.export(
        final_model,
        torch.randn(*input_shape),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
    )

    logger.info(
        "Successfully converted model from FastAI (%s) to ONNX (%s)"
        % (fastai_path, onnx_path)
    )
    return onnx_path


def torch2onnx(model, input_shape, onnx_path, input_names=None, output_names=None):
    """Convert PyTorch model to ONNX.

    Args:
        model (torch.nn.Module): PyTorch model.
        input_shape (tuple or list): Input shape (with batch dimension).
        onnx_path (str): Path to save converted model.
        input_names (tuple or list): Model's input names.
        output_names (tuple or list): Model's output names.
    """

    import torch

    torch.onnx.export(
        model,
        torch.randn(*input_shape),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
    )
    logger.info(
        "Successfully converted model from PyTorch (%s) to ONNX (%s)"
        % (model.__class__.__name__, onnx_path)
    )


def keras2onnx(model, onnx_path=None):
    """Convert Keras model to ONNX.

    Args:
        model (): Keras model.
        onnx_path (str): Path to save converted model.
    """

    import onnx
    import tf2onnx

    if onnx_path is None:
        onnx_path = model.name + ".onnx"

    onnx_model, _ = tf2onnx.convert.from_keras(model)
    onnx.save(onnx_model, onnx_path)
    logger.info(
        "Successfully converted model from Keras (%s) to ONNX (%s)"
        % (model.name, onnx_path)
    )


if __name__ == "__main__":

    def convert_mobilenet(weights_path):
        fastai2onnx(
            weights_path,
            input_shape=(1, 3, 224, 224),
            input_names=["image_1_3_224_224"],
            output_names=["look"],
        )

    mobilenet_path = "/home/rizhik/optimize_tensorrt/playground/weights/mobilenet.pkl"
    convert_mobilenet(mobilenet_path)
