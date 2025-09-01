from collections.abc import Callable
from typing import Any

import torch
from torch._ops import OpOverloadPacket
from torch._refs import is_complex
from torch.utils._pytree import tree_flatten, tree_unflatten

from ..complex_tensor import ComplexTensor

OpType = OpOverloadPacket

TableType = dict[OpType, Callable]
COMPLEX_OPS_TABLE: TableType = {}

COMPLEX_TO_REAL = {
    torch.complex128: torch.float64,
    torch.complex64: torch.float32,
    torch.complex32: torch.float16,
}

PROMOTE_TYPES_CPU = {
    torch.float16: torch.float32,
    torch.bfloat16: torch.float32,
}


def promote_real_cpu_tensors(
    tensor: torch.Tensor, *tensors: torch.Tensor
) -> tuple[torch.dtype, tuple[torch.Tensor, ...]]:
    out_dt = tensor.dtype
    for t in tensors:
        if isinstance(t, torch.Tensor):
            out_dt = torch.promote_types(out_dt, t.dtype)

    prom_dt = PROMOTE_TYPES_CPU.get(out_dt)
    if (
        prom_dt is None
        or tensor.device.type != "cpu"
        or any(t.device.type != "cpu" for t in tensors if isinstance(t, torch.Tensor))
    ):
        return out_dt, (
            tensor.to(out_dt),
            *(
                t.to(out_dt) if isinstance(t, torch.Tensor) else torch.asarray(t, dtype=out_dt)
                for t in tensors
            ),
        )

    return out_dt, (
        tensor.to(prom_dt),
        *(
            t.to(prom_dt) if isinstance(t, torch.Tensor) else torch.asarray(t, dtype=prom_dt)
            for t in tensors
        ),
    )


def register_complex(
    op: OpType,
    func_impl: Callable | None = None,
):
    """Decorator to register an implementation for some ops in some dispatch tables"""

    def inner(func):
        COMPLEX_OPS_TABLE[op] = func
        return func

    if func_impl is None:
        return inner
    return inner(func_impl)


FORCE_TEST_LIST: list[OpType] = []


def register_force_test(op: OpType, *args, **kwargs):
    FORCE_TEST_LIST.append(op)
    return register_complex(op, *args, **kwargs)


def lookup_complex(func, *args, **kwargs):
    return COMPLEX_OPS_TABLE.get(func, COMPLEX_OPS_TABLE.get(func.overloadpacket, None))


def split_complex_arg(
    arg: torch.Tensor | ComplexTensor | Any,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[Any, Any]:
    if isinstance(arg, ComplexTensor):
        return split_complex_tensor(arg)
    if isinstance(arg, torch.Tensor):
        if is_complex(arg):
            return arg.real, arg.imag
        return arg, torch.zeros_like(arg)
    if isinstance(arg, complex):
        return arg.real, arg.imag
    if isinstance(arg, float | torch.SymFloat):
        return arg, 0.0
    if isinstance(arg, int | torch.SymInt):
        return arg, 0
    if isinstance(arg, bool | torch.SymBool):
        return arg, False
    raise TypeError(f"Expected tensor or number got, {type(arg)}")


def split_complex_tensor(complex_tensor: ComplexTensor) -> tuple[torch.Tensor, torch.Tensor]:
    return complex_tensor.re, complex_tensor.im


def complex_to_real_dtype(dtype: torch.dtype) -> torch.dtype:
    return COMPLEX_TO_REAL.get(dtype, dtype)


def register_error(op: OpType):
    msg = f"`aten.{str(op).split('.', 1)[0]}` not implemented for `{ComplexTensor.__name__}`."

    exc_type = ERROR_TYPES.get(op, NotImplementedError)

    def ordered_impl(*args, **kwargs):
        raise exc_type(msg)

    func_name = f"{str(op).split('.', 1)}_impl"
    ordered_impl.__name__ = func_name
    ordered_impl.__qualname__ = func_name

    return register_force_test(op, ordered_impl)


ERROR_TYPES: dict[OpType, type[Exception]] = {}


def register_binary_nonlinear(op: OpType) -> Callable:
    def impl(lhs: ComplexTensor, rhs: ComplexTensor, *args, **kwargs) -> ComplexTensor:
        a_r, a_i = split_complex_tensor(lhs)
        b_r, b_i = split_complex_arg(rhs)
        out_dt, (a_r, a_i, b_r, b_i) = promote_real_cpu_tensors(a_r, a_i, b_r, b_i)
        real = op(a_r, b_r, *args, **kwargs) - op(a_i, b_i, *args, **kwargs)
        imag = op(a_r, b_i, *args, **kwargs) + op(a_i, b_r, *args, **kwargs)
        return ComplexTensor(real.to(out_dt), imag.to(out_dt))

    func_name = f"{str(op).split('.', 1)}_impl"
    impl.__name__ = func_name
    impl.__qualname__ = func_name

    return register_complex(op, impl)


def register_simple(op: OpType):
    def impl(self: ComplexTensor, *args, **kwargs) -> ComplexTensor:
        x, y = split_complex_tensor(self)
        u = op(x, *args, **kwargs)
        v = op(y, *args, **kwargs)
        u_flat, u_spec = tree_flatten(u)
        v_flat, v_spec = tree_flatten(v)
        assert u_spec == v_spec
        out_flat = [ComplexTensor(ui, vi) for ui, vi in zip(u_flat, v_flat, strict=False)]
        return tree_unflatten(out_flat, u_spec)

    func_name = f"{str(op).split('.', 1)}_impl"
    impl.__name__ = func_name
    impl.__qualname__ = func_name

    return register_complex(op, impl)
