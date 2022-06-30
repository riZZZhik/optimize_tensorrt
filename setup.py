from setuptools import setup, find_packages

setup(
    name="optimize_tensorrt",
    version="1.0.1",
    description="Convert and inference TensorRT models",
    license="Apache",
    author="Dmtry Barsukov",
    author_email="riZZZhik@gmail.com",
    packages=find_packages("src"),
    package_dir={"": "src"},
    url="https://github.com/riZZZhik/optimize_tensorrt",
    install_requires=["numpy", "loguru"],
)
