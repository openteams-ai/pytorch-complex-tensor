from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

import torch
from torch._decomp import get_decompositions
from torch._ops import OpOverload, OpOverloadPacket
from torch._refs import is_complex as _is_complex
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from typing_extensions import Self

from ..complex_tensor import ComplexTensor

OpType = OpOverloadPacket

TableType = dict[OpType, Callable]
COMPLEX_OPS_TABLE: TableType = {}

COMPLEX_TO_REAL = {
    torch.complex128: torch.float64,
    torch.complex64: torch.float32,
    torch.complex32: torch.float16,
}

REAL_TO_COMPLEX = {v: k for k, v in COMPLEX_TO_REAL.items()}

PROMOTE_TYPES_CPU = {
    torch.float16: torch.float32,
    torch.bfloat16: torch.float32,
}


def promote_real_cpu_tensors(
    *tensors: torch.Tensor,
) -> tuple[torch.dtype, tuple[torch.Tensor, ...]]:
    tensor = next(t for t in tensors if isinstance(t, torch.Tensor))
    out_dt = tensor.dtype
    for t in tensors:
        if isinstance(t, torch.Tensor):
            out_dt = torch.promote_types(out_dt, t.dtype)

    prom_dt = PROMOTE_TYPES_CPU.get(out_dt)
    if prom_dt is None or any(
        t.device.type != "cpu" for t in tensors if isinstance(t, torch.Tensor)
    ):
        return out_dt, tuple(
            t.to(out_dt) if isinstance(t, torch.Tensor) else torch.asarray(t, dtype=out_dt)
            for t in tensors
        )

    return out_dt, tuple(
        t.to(prom_dt) if isinstance(t, torch.Tensor) else torch.asarray(t, dtype=prom_dt)
        for t in tensors
    )


def register_complex(
    op: OpType,
    func_impl: Callable | None = None,
):
    """Decorator to register an implementation for some ops in some dispatch tables"""

    def inner(func):
        if COMPLEX_OPS_TABLE.get(op, func) is not func:
            raise RuntimeError(
                "Attempted to register multiple functions for "
                f"{op._qualified_op_name.replace('::', '.')}"
            )
        COMPLEX_OPS_TABLE[op] = func
        return func

    if func_impl is None:
        return inner
    return inner(func_impl)


FORCE_TEST_LIST: list[OpType] = []


def register_force_test(op: OpType, *args, **kwargs):
    FORCE_TEST_LIST.append(op)
    return register_complex(op, *args, **kwargs)


DECOMPOSITIONS = get_decompositions(list(torch.ops.aten))
DEBUG_SET: ContextVar[set[OpType] | None] = ContextVar("DEBUG_SET", default=None)


def lookup_complex(func: OpType, *args, **kwargs) -> Callable:
    return COMPLEX_OPS_TABLE.get(
        func,
        COMPLEX_OPS_TABLE.get(
            func.overloadpacket, DECOMPOSITIONS.get(func, DECOMPOSITIONS.get(func.overloadpacket))
        ),
    )


def is_complex(x: torch.Tensor, /) -> bool:
    return (isinstance(x, torch.Tensor) and _is_complex(x)) or isinstance(x, complex)


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


def _get_op_name(op: OpType) -> str:
    return str(op).split(".", 1)[1]


def _get_func_name(op: OpType) -> str:
    return f"{_get_op_name(op)}_impl"


def register_error(op: OpType):
    msg = f"`aten.{_get_op_name(op)}` not implemented for `{ComplexTensor.__name__}`."

    exc_type = ERROR_TYPES.get(op, NotImplementedError)

    def ordered_impl(*args, **kwargs):
        raise exc_type(msg)

    func_name = _get_func_name(op)
    ordered_impl.__name__ = func_name
    ordered_impl.__qualname__ = func_name

    return register_force_test(op, ordered_impl)


ERROR_TYPES: dict[OpType, type[Exception]] = {}


def register_binary_nonlinear(op: OpType) -> Callable:
    def impl(lhs: ComplexTensor, rhs: ComplexTensor, *args, **kwargs) -> ComplexTensor:
        a_r, a_i = split_complex_arg(lhs)
        b_r, b_i = split_complex_arg(rhs)
        out_dt, (a_r, a_i, b_r, b_i) = promote_real_cpu_tensors(a_r, a_i, b_r, b_i)
        real = op(a_r, b_r, *args, **kwargs) - op(a_i, b_i, *args, **kwargs)
        imag = op(a_r, b_i, *args, **kwargs) + op(a_i, b_r, *args, **kwargs)
        return ComplexTensor(real.to(out_dt), imag.to(out_dt))

    func_name = _get_func_name(op)
    impl.__name__ = func_name
    impl.__qualname__ = func_name

    return register_complex(op, impl)


def register_simple(op: OpType):
    def impl(
        self: ComplexTensor, *args, dtype: torch.dtype | None = None, **kwargs
    ) -> ComplexTensor:
        x, y = split_complex_tensor(self)
        if dtype is not None and dtype not in COMPLEX_TO_REAL:
            raise RuntimeError("Non-complex `dtype` specified, please write custom impl.")

        if dtype in COMPLEX_TO_REAL:
            kwargs["dtype"] = COMPLEX_TO_REAL[dtype]

        u = op(x, *args, **kwargs)
        v = op(y, *args, **kwargs)

        u_flat, u_spec = tree_flatten(u)
        v_flat, v_spec = tree_flatten(v)
        assert u_spec == v_spec
        out_flat = [ComplexTensor(ui, vi) for ui, vi in zip(u_flat, v_flat, strict=False)]
        return tree_unflatten(out_flat, u_spec)

    func_name = _get_func_name(op)
    impl.__name__ = func_name
    impl.__qualname__ = func_name

    return register_complex(op, impl)


def _as_complex_tensor(arg: torch.Tensor | Any) -> torch.Tensor | ComplexTensor | Any:
    if (
        not isinstance(arg, ComplexTensor)
        and isinstance(arg, torch.Tensor)
        and arg.dtype in COMPLEX_TO_REAL
    ):
        return ComplexTensor.from_interleaved(arg)
    return arg


def _as_interleaved(arg: ComplexTensor | Any) -> torch.Tensor | Any:
    if isinstance(arg, ComplexTensor):
        return arg.as_interleaved()
    return arg


class ComplexDispatchMode(TorchDispatchMode):
    def __init__(self, _dispatch_key=None, *, _compile: bool = False, _debug: bool = False):
        super().__init__(_dispatch_key)
        self._compile = _compile
        self._debug = _debug
        self._debug_token = None

    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: tuple[type],
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ):
        if kwargs is None:
            kwargs = {}

        if self._compile:
            func = torch.compile(func)

        args = tree_map(_as_complex_tensor, args)
        kwargs = tree_map(_as_complex_tensor, kwargs)

        return tree_map(_as_interleaved, func(*args, **kwargs))

    def __enter__(self) -> Self:
        # Note (debugging ops): This block sets the debugging mode
        if self._debug:
            self._debug_token = DEBUG_SET.set(set())
        return super().__enter__()

    def __exit__(self, type_, val, tb):
        # Note (debugging ops): This block resets the debugging mode
        if self._debug_token is not None:
            print("\n".join([str(op) for op in DEBUG_SET.get()]))
            DEBUG_SET.reset(self._debug_token)
            self._debug_token = None
        return super().__exit__(type_, val, tb)
