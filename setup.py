import glob
import os
from pathlib import Path
from subprocess import run

import torch
import torch.cuda
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

# from torchsparse import __version__

MAX_JOBS = os.getenv("MAX_JOBS")
need_to_unset_max_jobs = False
if not MAX_JOBS:
    need_to_unset_max_jobs = True
    os.environ["MAX_JOBS"] = "10"
    print(f"Setting MAX_JOBS: {os.environ['MAX_JOBS']}")

version_file = open("./torchsparse/version.py")
version = version_file.read().split("'")[1]
print("torchsparse version:", version)

if (torch.cuda.is_available() and CUDA_HOME is not None) or (
    os.getenv("FORCE_CUDA", "0") == "1"
):
    device = "cuda"
    pybind_fn = f"pybind_{device}.cu"
else:
    device = "cpu"
    pybind_fn = f"pybind_{device}.cpp"

sources = [os.path.join("torchsparse", "backend", pybind_fn)]
for fpath in glob.glob(os.path.join("torchsparse", "backend", "**", "*")):
    if (fpath.endswith("_cpu.cpp") and device in ["cpu", "cuda"]) or (
        fpath.endswith("_cuda.cu") and device == "cuda"
    ):
        sources.append(fpath)

extension_type = CUDAExtension if device == "cuda" else CppExtension
current_dir = Path(__file__).parent.resolve()
sparsehash_dir = current_dir / "torchsparse" / "backend" / "third_party" / "sparsehash"
sparsehash_dir_inc = sparsehash_dir / "src"
sparseconfig_path = sparsehash_dir_inc / "sparsehash" / "internal" / "sparseconfig.h"

if not sparseconfig_path.exists():
    print("Generating sparseconfig.h ...")
    run(["./configure"], cwd=sparsehash_dir, check=True)
    run(["make", "src/sparsehash/internal/sparseconfig.h"], cwd=sparsehash_dir, check=True)

extra_compile_args = {
    "cxx": ["-O3", "-fopenmp", "-lgomp", f"-I{sparsehash_dir_inc}"],
    "nvcc": ["-O3"],
}

setup(
    name="torchsparse",
    version=version,
    packages=find_packages(),
    ext_modules=[
        extension_type(
            "torchsparse.backend", sources, extra_compile_args=extra_compile_args
        )
    ],
    url="https://github.com/mit-han-lab/torchsparse",
    install_requires=[
        "numpy",
        "backports.cached_property",
        "tqdm",
        "typing-extensions",
        "wheel",
        "rootpath",
        "torch",
        "torchvision"
    ],
    dependency_links=[
        'https://download.pytorch.org/whl/cu118'
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)

if need_to_unset_max_jobs:
    print("Unsetting MAX_JOBS")
    os.environ.pop("MAX_JOBS")

# Cleanup sparsehash configure/make artifacts
run(["git", "clean", "-fd"], cwd=sparsehash_dir, check=False)
