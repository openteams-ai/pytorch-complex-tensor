import torch

from ._common import (
    register_force_test,
    register_simple,
)

_c10d_functional = torch.ops._c10d_functional

# TODO (hameerabbasi): Not being tested
broadcast_impl = register_force_test(
    _c10d_functional.broadcast, register_simple(_c10d_functional.broadcast)
)

# TODO (hameerabbasi): Not being tested
broadcast__impl = register_force_test(
    _c10d_functional.broadcast_, register_simple(_c10d_functional.broadcast_)
)
