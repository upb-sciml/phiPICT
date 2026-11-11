Installation
============

phiPICT ships a compiled CUDA extension (``phipict._C``). It requires Linux, an
NVIDIA GPU, and Python 3.11-3.13. We recommend a dedicated virtual environment.

1. From PyPI
------------

.. code-block:: bash

    pip install phipict

This installs the default build together with the PyTorch it was built for:
PyTorch 2.10 with CUDA 12.8, which needs an NVIDIA driver ≥ 570. pyamg, used for
the AMG pressure solver, is installed as well.

There is no need to install PyTorch first. If a different PyTorch version is
installed, pip replaces it with 2.10, because the extension only works with the
PyTorch version it was built against (see below). To use another PyTorch or
CUDA version, build from source (section 2).

Two optional extras exist, mainly for contributors: ``dev`` (linters, type
checking, tests) and ``docs`` (Sphinx), e.g. ``pip install "phipict[dev]"``.
Reading VTK formats other than ``.vts``/``.vtm`` needs pyvista
(``pip install pyvista``).

Why PyTorch 2.10?
-----------------

The extension is bound to two versions it was built against:

- the **PyTorch minor version** (e.g. 2.12), because libtorch has no stable C++ ABI;
- the **CUDA major version** (cu12x or cu13x). A build for CUDA 13.0 also works
  with PyTorch built for CUDA 13.2.

The PyPI wheel is built for PyTorch 2.10 with CUDA 12.8, the newest combination
that runs on NVIDIA drivers ≥ 570. For any other combination, build from source.
Check your installed PyTorch with:

.. code-block:: bash

    python -c "import torch; print(torch.__version__, torch.version.cuda)"

2. Building from source
-----------------------

phiPICT builds against the PyTorch already installed in your environment, so
any PyTorch ≥ 2.9 with a CUDA build works. This requires a CUDA toolkit whose major
version matches ``torch.version.cuda`` (``nvcc`` on the ``PATH`` or
``CUDA_HOME`` set) and ``setuptools>=77``, since the build uses the environment's
PyTorch and setuptools. With conda, ``conda install -c conda-forge cuda-toolkit
cuda-version=12.8`` also installs a gcc/g++ that this nvcc supports.

.. code-block:: bash

    git clone https://github.com/upb-sciml/phiPICT.git
    cd phiPICT
    make install        # builds and installs the package
    make install-dev    # editable install with development tools

The build honours the usual PyTorch extension variables:

- ``TORCH_CUDA_ARCH_LIST``: target GPU architectures, ``"8.0;8.6;9.0+PTX"`` in
  the Makefile. Add ``12.0`` for Blackwell GPUs (needs CUDA ≥ 12.8).
- ``MAX_JOBS``: number of parallel compile jobs. Compiling is memory hungry, so
  the Makefile uses ``1``.
- ``PHIPICT_BUILD_NOISE_EXT=1``: additionally build the simplex noise extension
  ``phipict._noise`` (``make install-dev`` does this).

Troubleshooting
---------------

An ``ImportError: ... undefined symbol`` when importing ``phipict`` means that the
installed wheel was built for a different PyTorch version, e.g. because PyTorch
was upgraded after installing phipict. Reinstall phipict from PyPI (which
reinstalls PyTorch 2.10), or build it from source against your PyTorch.
