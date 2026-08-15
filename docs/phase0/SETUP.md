# ASRI Phase-0 Environment Setup (Windows / NVIDIA)

Bounded note recording how the Phase-0 smoke environment was actually built on
the execution machine. This is a reproducibility record, not a general
installation guide, and it does not change the project's dependency contract.

## Observed constraint

On this machine (Windows 11 Pro, RTX 3080 Ti, NVIDIA driver 610.47), the default
PyPI resolution of the project's `torch>=2.3` dependency produced a **CPU-only**
build (`torch==2.13.0+cpu`, `torch.cuda.is_available() == False`), because the
default `win_amd64` wheel on PyPI carries no CUDA runtime. The frozen Phase-0
path then falls back to CPU/float32, which does not exercise the intended
GPU substrate.

## Working installation

The successful environment used:

```text
Python          3.12.10
torch           2.13.0+cu130   (from the official PyTorch CUDA index)
transformers    5.15.0
NVIDIA driver   610.47         (CUDA UMD 13.3)
```

The CUDA-enabled PyTorch wheel **must be installed from the official PyTorch
CUDA index before the editable project install**, otherwise pip/uv will keep
the CPU-only wheel already present:

```bash
python -m venv .venv
uv pip install --python .venv/Scripts/python.exe \
    --index-url https://download.pytorch.org/whl/cu130 "torch==2.13.0+cu130"
uv pip install --python .venv/Scripts/python.exe -e .
```

(`pip install` works equivalently; on this network the pip single-connection
download stalled while `uv` completed, which is why `uv` is shown.)

Note: no `cu128` wheel of torch 2.13.0 exists for Windows; `cu130` is the
variant that matched both the available wheels and the installed driver.
