<p align="center">
    <a href="./docs/source/_static/img/logo_lm.png#gh-light-mode-only">
        <img src="./docs/source/_static/img/logo_lm.png#gh-light-mode-only" alt="phiPICT Logo" width="40%"/>
    </a>
    <a href="./docs/source/_static/img/logo_dm.png#gh-dark-mode-only">
        <img src="./docs/source/_static/img/logo_dm.png#gh-dark-mode-only" alt="phiPICT Logo" width="40%"/>
    </a>
</p>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.8-%2376B900)
![License](https://img.shields.io/badge/License-Apache--2.0-orange)
[![Release](https://github.com/upb-sciml/phiPICT/actions/workflows/release.yml/badge.svg)](https://github.com/upb-sciml/phiPICT/actions/workflows/release.yml)

</div>

<h1 align="center">φ-PICT: Differentiable GPU-Accelerated Solver for Hydrodynamic &amp; Magnetohydrodynamic Flows</h1>

---

## Key Features

- **Differentiable Physics:** Fully end-to-end differentiable solver supporting Backpropagation Through Time (BPTT/APG), Short-Horizon Actor-Critic (SHAC) and more.
- **Massively Batched Execution:** Efficiently parallelize over 100 of independent fluid environments on a single GPU.
- **MHD & Standard Hydrodynamics:** Simulates both magnetohydrodynamic (MHD) flows and standard hydrodynamic fluid mechanics.
- **RL Framework Integration:** Seamlessly integrated into [FluidGym](https://github.com/upb-sciml/fluidgym) for plug-and-play benchmarking of control algorithms.

## Installation

phiPICT needs Linux, an NVIDIA GPU with a driver for CUDA 12.8 (≥ 570) and
Python 3.11–3.13.

### 📦 Installation from PyPI

```bash
pip install phipict
```

This also installs the matching PyTorch (2.10, built with CUDA 12.8), pyamg, and
the remaining runtime dependencies (NumPy, SciPy, matplotlib, imageio, psutil).
There is no need to install PyTorch first: the compiled extension only works with
the PyTorch version it was built against, so pip replaces any other version.
Wheels for other PyTorch/CUDA versions are listed in the
[installation docs](https://safe-autonomous-systems.github.io/phiPICT/installation.html).

Optional extras:

```bash
pip install "phipict[dev]"   # linters, type checking, and tests
pip install "phipict[docs]"  # Sphinx, to build the documentation
```

Reading VTK formats other than `.vts`/`.vtm` needs `pip install pyvista`.

### 🧱 Build from Source

Building from source compiles the CUDA kernels against the PyTorch installed in
your environment. Here is an example for Python 3.12 and CUDA 12.8.

1. Create a new conda environment and activate it:
```bash
conda create -n phipict python=3.12
conda activate phipict
```

2. Install the CUDA toolkit from conda-forge. This also installs a gcc/g++ that
   the chosen nvcc supports, so no separate compiler install is needed. The CUDA
   major version must match the one your PyTorch build uses (12 here):
```bash
conda install -c conda-forge pip cuda-toolkit cuda-version=12.8
conda env config vars set CUDA_HOME="$CONDA_PREFIX"
conda activate phipict  # re-activate so CUDA_HOME takes effect
```

3. Install PyTorch for the same CUDA version via pip. Any version ≥ 2.9 works;
   the extension is then bound to this version:
```bash
pip install "torch==2.10.*" --index-url https://download.pytorch.org/whl/cu128
pip install "setuptools>=77"  # the build uses the environment's setuptools
```

4. Clone the repository and enter the directory, then compile the custom CUDA
   kernels and install the package (this might take several minutes):
```bash
make install
```

For development, use `make install-dev` instead, which installs the package in
editable mode together with the development dependencies.

The documentation dependencies are installed via `pip install ".[docs]"`.

## Getting Started

See the [`examples`](examples) directory for example simulations.

## License

This repository is published under the Apache-2.0 license.
