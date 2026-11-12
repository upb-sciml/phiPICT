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
PyTorch version it was built against (see below). For another PyTorch or CUDA
version, use a wheel from the wheel index (section 2) or build from source
(section 3).

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
that runs on NVIDIA drivers ≥ 570. Other combinations are available from the
wheel index (section 2) or can be built from source (section 3). Check your
installed PyTorch with:

.. code-block:: bash

    python -c "import torch; print(torch.__version__, torch.version.cuda)"

On import, phiPICT checks that the installed PyTorch matches the one it was
built for and otherwise raises an ``ImportError`` that names the matching wheel.

2. Other PyTorch versions: wheel index
--------------------------------------

Prebuilt wheels for further PyTorch/CUDA combinations are published with every
release on our `wheel index <https://upb-sciml.github.io/phiPICT/whl/>`_:

=======  ======================================  ======================
PyTorch  CUDA 12 (driver ≥ 570)                  CUDA 13 (driver ≥ 580)
=======  ======================================  ======================
2.10     PyPI default (``pip install phipict``)  –
2.13     –                                       ✓
2.14     ✓                                       ✓
2.15     –                                       ✓
=======  ======================================  ======================

There is one page per PyTorch minor version and CUDA major version,
``torch-<PyTorch minor>+cu<CUDA major>.html``. A wheel for CUDA 13 works with
every CUDA 13 build of PyTorch (cu130, cu132, ...), and a wheel for CUDA 12 with
cu126 and cu129. Install PyTorch first, then phiPICT from the matching page,
e.g. for PyTorch 2.14 with CUDA 12.6:

.. code-block:: bash

    pip install "torch==2.14.*" --index-url https://download.pytorch.org/whl/cu126
    pip install phipict -f https://upb-sciml.github.io/phiPICT/whl/torch-2.14+cu12.html

To pick the page for the PyTorch already installed in your environment:

.. code-block:: bash

    pip install phipict -f "https://upb-sciml.github.io/phiPICT/whl/torch-$(python -c 'import torch; v = torch.__version__.split("."); print(f"{v[0]}.{v[1]}+cu{torch.version.cuda.split(".")[0]}")').html"

The wheel pins its PyTorch minor version (e.g. ``torch==2.14.*``), so pip keeps
the installed PyTorch. The wheels for CUDA 12 are compiled with CUDA 12.6, which
does not support Blackwell GPUs natively; they run there through PTX JIT
compilation. On Blackwell, prefer the CUDA 13 wheels.

The index covers the newest PyTorch versions and is updated with each release.
For combinations it does not cover, build from source.

3. Building from source
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

An ``ImportError`` saying that phiPICT was built for a different PyTorch means
that PyTorch was changed after installing phipict, e.g. upgraded. Install the
matching wheel from the wheel index as shown in the error message, reinstall
phipict from PyPI (which reinstalls PyTorch 2.10), or build it from source
against your PyTorch. Versions of phipict before 0.1.1 show this as
``ImportError: ... undefined symbol`` instead.
