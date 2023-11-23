#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import glob
import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 3], "Requires PyTorch >= 1.3"


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "adet", "layers", "csrc")

    main_source = os.path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu")) + glob.glob(
        os.path.join(extensions_dir, "*.cu")
    )

    sources = [main_source] + sources

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv(
        "FORCE_CUDA", "0"
    ) == "1":
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

        # It's better if pytorch can do this by default ..
        CC = os.environ.get("CC", None)
        if CC is not None:
            extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "stoma._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="StomaAI",
    version="0.1.1",
    author="JPB",
    url="https://github.com/XDynames/SAI-training",
    description="SAI is a joint collaboration between"
    "The University of Adelaide's Australian Institute for"
    " Machine Learning and Plant Energy Biology ARC Center"
    " of Excellence.",
    packages=find_packages(exclude=("configs", "tests")),
    python_requires=">=3.6",
    install_requires=[
        "termcolor>=1.1",
        "Pillow>=6.0",
        "yacs>=0.1.6",
        "tabulate",
        "cloudpickle",
        "matplotlib",
        "mask-to-polygons",
        "tqdm>4.29.0",
        "tensorboard",
        "shapely==1.6.4",
        "xmltodict",
        "iopath==0.1.6",
        "tabulate==0.8.9",
        "scipy",
    ],
    extras_require={"all": ["psutil"]},
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
