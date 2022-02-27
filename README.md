# Всем привет )
Этот код нужен что-бы оптимизировать модель в TensorRT и инференсить в лайве.

Важный момент: engine файлы не совместимы между системами, нужно оптимизировать под каждую систему отдельно.


# Recommended pipeline
- Заранее перегнать модель из фреймворка в ONNX. Для этого есть несколько функций в этом модуле. `python -c "import onnx_converter; help(onnx_converter)"`
- Скачать эту либу и onnx файл модели на необходимое железо.
- Поставить все необходимые зависимости
- Указать все пути для энвов из пункта _Installation_
- Запустить оптимизатор onnx2trt: `python onnx2trt.py --help`
  - Для INT8 необходимо заранее создать calibration файл через onnx2trt_handmade. `python -c "from onnx2trt_handmade import onnx2trt_handmade; help(onnx2trt_handmade)"`
- Запомнить куда сохранился итоговый engine файл.
- Для инференса использовать класс InferenceTRT. `python -c "from inference import InferenceTRT; help(InferenceTRT)"`


# Installation

## Envs
- `CUDA_HOME`: Путь до папки с кудой.
- `LD_LIBRARY_PATH`:
  - Путь до `lib64` в папке с кудой.
  - Путь до `targets/x86_64-linux/lib/` в папке с кудой.
  - Путь до `lib` в папке с TensorRT.

## `trtexec`
Для запуска onnx2trt необходимо указывать путь до trtexec. Как правило, он хранится по пути `/targets/x86_64-linux-gnu/bin/trtexec` в папке с TensorRT.

## `TensorRT`
Tested on version `8.2.2`

## `CUDA`

## `PyCUDA`


# Convert to ONNX

## `fastai2onnx`
Конвертируем модельку из FastAI в ONNX.

### Args:
- `fastai_path` (str): Path to fastai model file.
- `input_shape` (tuple or list): Input shape (with batch dimension).
- `onnx_path` (str): Path to save converted model.
- `input_names` (tuple or list): Model's input names.
- `output_names` (tuple or list): Model's output names.

## `torch2onnx`
Конвертируем модельку из PyTorch в ONNX.

### Args:
- `model` (torch.nn.Module): PyTorch model.
- `input_shape` (tuple or list): Input shape (with batch dimension).
- `onnx_path` (str): Path to save converted model.
- `input_names` (tuple or list): Model's input names.
- `output_names` (tuple or list): Model's output names.

## `keras2onnx`
Конвертируем модельку из Keras в ONNX.

### Args:
- `model` (): Keras model.
- `input_shape` (tuple or list): Input shape (with batch dimension).
- `onnx_path` (str): Path to save converted model.


# Optimize

## `onnx2trt`
Самый оптимальный метод для оптимизации модельки под TensorRT. В конце конвертации выводит бенчмарки. Не поддерживает калибровку для INT8. 

_NB! Для каждой машины нужно конвертировать отдельно, просто перетащить файлик engine не сработает._

### Args:
- `onnx_path` (str): Path to onnx model (opset9).
- `engine_path` (str): Path to save engine.
- `dtype` (str or np.dtype): Model data type.
- `use_sparsity` (bool): Use sparsity boolean.
- `trtexec_path` (str): Path to trtexec binary.
- `workspace_size` (int): Size of workspace in MB.

## `onnx2trt_handmade`
Оптимизация модельки с использованием TensorRT Python API.
Лучше всего использовать один раз для создания calibration файлика, его можно переносить между системами. И под каждую оптимизировать через trtexec. 

### Args:
- `onnx_path` (str): Path to ONNX model.
- `engine_path` (str): Path to save TensorRT engine.
- `dtype` (str): Model data type. (fp32, fp16 or int8).
- `calib` (trt.IInt8EntropyCalibrator2): Calibration class. Only for INT8.

## `Calibration`
Класс для агрегации данных во время калибровки INT8.

### `__init__`
#### Args:
- `calibration_files` (list or tuple): List of paths to images for calibration.
- `preprocessor` (function): Image preprocessing function.
- `shape` (list or tuple): Input image shape. With batch_size.
- `cache_file` (str): Path to save calibration.


# Inference

## InferenceTRT
Класс для инференса модельки оптимизированной в engine.

### `__init__`
Initialize class variables.
#### Args:
- `engine_path` (str): Path to TensorRT engine file.
- `device` (cuda.Device): Cuda device to run inference on.

### `load_engine`
Load TensorRT engine from file.
#### Args:
- `engine_path` (str): Path to TensorRT engine file.

### `predict`
Get model prediction.
#### Args:
- `inputs` (np.ndarray): Model inputs array.
#### Returns:
- Model prediction as np.ndarray.