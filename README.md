# `complex-tensor`

Subclass of `torch.Tensor` for working with complex numbers.

# Development Setup

For now, the development setup uses the CPU version of PyTorch only.

1. [Install `uv`](https://docs.astral.sh/uv/getting-started/installation/)
2. Run the tests with `uv run pytest -n auto` to get started.
   * This will also create a virtual environment in `.venv/`.

# Repository Structure

* The main `torch.Tensor` subclass is found in [`src/complex_tensor/complex_tensor.py`](https://github.com/openteams-ai/pytorch-complex-tensor/blob/main/src/complex_tensor/complex_tensor.py).
* Operations are implemented in the [`src/complex_tensor/ops/`](https://github.com/openteams-ai/pytorch-complex-tensor/tree/main/src/complex_tensor/ops) directory.
  * [`_common.py`](https://github.com/openteams-ai/pytorch-complex-tensor/blob/main/src/complex_tensor/ops/_common.py) defines some basic utility functions.
  * [`aten.py`](https://github.com/openteams-ai/pytorch-complex-tensor/blob/main/src/complex_tensor/ops/aten.py) defines overloads for `torch.ops.aten`.
  * [`prims.py`](https://github.com/openteams-ai/pytorch-complex-tensor/blob/main/src/complex_tensor/ops/aten.py) does the same for `torch.ops.prims`.
   * Currently, this directory is empty.
* Tests are located in [`src/complex_tensor/test`](https://github.com/openteams-ai/pytorch-complex-tensor/tree/main/src/complex_tensor/test).
  * Testing currently needs to be expanded; currently only tests which provide `OpInfo`s in `torch.testing._internal.common_methods_invocations.op_db` are tested.
  * Exceptions are noted in-tree with a `TODO`.
  * A warning is emitted during `pytest` noting the missing ops.

This repository is currently WIP, which means not all ops are implemented, but many common ones are.
