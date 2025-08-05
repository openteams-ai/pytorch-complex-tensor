from typing import Any, Callable, Optional, Union

import torch
from torch._ops import OpOverloadPacket
from torch._refs import is_complex

from complex_tensor import ComplexTensor

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

aten = torch.ops.aten


def register_complex(
    op: OpType,
    func_impl: Optional[Callable] = None,
):
    """Decorator to register an implementation for some ops in some dispatch tables"""

    def inner(func):
        COMPLEX_OPS_TABLE[op] = func
        return func

    if func_impl is None:
        return inner
    return inner(func_impl)


def lookup_complex(func, *args, **kwargs):
    return COMPLEX_OPS_TABLE.get(func, COMPLEX_OPS_TABLE.get(func.overloadpacket, None))


def split_complex_arg(
    arg: Union[torch.Tensor, ComplexTensor, Any],
) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[Any, Any]]:
    if isinstance(arg, ComplexTensor):
        return split_complex_tensor(arg)
    if isinstance(arg, torch.Tensor):
        if is_complex(arg):
            return arg.real, arg.imag
        return arg, torch.zeros_like(arg)
    if isinstance(arg, complex):
        return arg.real, arg.imag
    if isinstance(arg, (float, torch.SymFloat)):
        return arg, 0.0
    if isinstance(arg, (int, torch.SymInt)):
        return arg, 0
    if isinstance(arg, (bool, torch.SymBool)):
        return arg, False
    raise TypeError(f"Expected tensor or number got, {type(arg)}")


def split_complex_tensor(complex_tensor: ComplexTensor) -> tuple[torch.Tensor, torch.Tensor]:
    return complex_tensor.re, complex_tensor.im


def complex_to_real_dtype(dtype: torch.dtype) -> torch.dtype:
    return COMPLEX_TO_REAL.get(dtype, dtype)


# Not sure why torch dispatch does not hit here.
@register_complex(aten.real)
def real(self: ComplexTensor) -> torch.Tensor:
    re, _ = split_complex_tensor(self)
    return re


# Not sure why torch dispatch does not hit here.
@register_complex(aten.imag)
def imag(self: ComplexTensor) -> torch.Tensor:
    _, im = split_complex_tensor(self)
    return im


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


def register_binary_nonlinear(aten_op: OpType) -> Callable:
    def impl(lhs: ComplexTensor, rhs: ComplexTensor, *args, **kwargs) -> ComplexTensor:
        a_r, a_i = split_complex_tensor(lhs)
        b_r, b_i = split_complex_arg(rhs)
        out_dt, (a_r, a_i, b_r, b_i) = promote_real_cpu_tensors(a_r, a_i, b_r, b_i)
        real = aten_op(a_r, b_r, *args, **kwargs) - aten_op(a_i, b_i, *args, **kwargs)
        imag = aten_op(a_r, b_i, *args, **kwargs) + aten_op(a_i, b_r, *args, **kwargs)
        return ComplexTensor(real.to(out_dt), imag.to(out_dt))

    return register_complex(aten_op, impl)


def register_binary_linear(aten_op):
    def impl_with_alpha(
        lhs: ComplexTensor, rhs: ComplexTensor, *args, alpha, **kwargs
    ) -> ComplexTensor:
        return aten_op(lhs, aten.mul(rhs, alpha, *args, **kwargs), *args, **kwargs)

    def impl(lhs: ComplexTensor, rhs: ComplexTensor, *args, **kwargs) -> ComplexTensor:
        alpha = kwargs.pop("alpha", None)
        if alpha is not None:
            return impl_with_alpha(lhs, rhs, *args, alpha=alpha, **kwargs)
        a_r, a_i = split_complex_tensor(lhs)
        b_r, b_i = split_complex_arg(rhs)
        out_dt, (a_r, a_i, b_r, b_i) = promote_real_cpu_tensors(a_r, a_i, b_r, b_i)
        u = aten_op(a_r, b_r, *args, **kwargs)
        v = aten_op(a_i, b_i, *args, **kwargs)
        return ComplexTensor(u.to(out_dt), v.to(out_dt))

    return register_complex(aten_op, impl)


def _make_simple(aten_op: OpType):
    def impl(self: ComplexTensor, *args, **kwargs) -> ComplexTensor:
        x, y = split_complex_tensor(self)
        u = aten_op(x, *args, **kwargs)
        v = aten_op(y, *args, **kwargs)
        return ComplexTensor(u, v)

    return impl


def register_simple(aten_op: OpType):
    return register_complex(aten_op, _make_simple(aten_op))


slice_impl = register_simple(aten.slice)

# some binary ops which we can stamp out
mul_impl = register_binary_nonlinear(aten.mul)
mm_impl = register_binary_nonlinear(aten.mm)
add_impl = register_binary_linear(aten.add)
sub_impl = register_binary_linear(aten.sub)


@register_complex(aten.div)
def div(lhs: ComplexTensor, rhs: ComplexTensor, *, rounding_mode=None):
    a_r, a_i = split_complex_tensor(lhs)
    b_r, b_i = split_complex_arg(rhs)
    out_dt, (a_r, a_i, b_r, b_i) = promote_real_cpu_tensors(a_r, a_i, b_r, b_i)
    num_r = a_r * b_r + a_i * b_i
    num_i = a_i * b_r - a_r * b_i
    den = b_r * b_r + b_i * b_i
    return ComplexTensor(
        aten.div(num_r, den, rounding_mode=rounding_mode).to(out_dt),
        aten.div(num_i, den, rounding_mode=rounding_mode).to(out_dt),
    )


# reductions
@register_complex(aten.prod)
def prod_impl(self: ComplexTensor, *args, **kwargs) -> ComplexTensor:
    dtype = kwargs.pop("dtype", self.dtype)
    kwargs["dtype"] = complex_to_real_dtype(dtype)

    prod_r = torch.prod(torch.abs(self), *args, **kwargs)
    sum_phi = torch.sum(torch.angle(self), *args, **kwargs)
    u = prod_r * torch.cos(sum_phi)
    v = prod_r * torch.sin(sum_phi)
    return ComplexTensor(u, v)


@register_complex(aten.sum)
def sum_impl(self: ComplexTensor, *args, **kwargs) -> ComplexTensor:
    re, im = split_complex_tensor(self)
    return ComplexTensor(torch.sum(re, *args, **kwargs), torch.sum(im, *args, **kwargs))


# unary funcs,
# most of these are simple or require some kind of identity
@register_complex(aten.abs)
def abs_impl(self: ComplexTensor) -> torch.Tensor:
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_real_cpu_tensors(x, y)
    scale = torch.maximum(torch.abs(x), torch.abs(y))
    result = torch.where(
        scale.to(torch.bool),
        torch.sqrt(torch.pow(x / scale, 2) + torch.pow(y / scale, 2)) * scale,
        False,
    )
    return result.to(out_dt)


@register_complex(aten.angle)
def angle_impl(self: ComplexTensor) -> torch.Tensor:
    x, y = split_complex_tensor(self)
    return torch.arctan2(y, x)


@register_complex(aten.acos)
def acos_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_real_cpu_tensors(x, y)

    y2 = y**2
    a = (x**2) + y2
    b = torch.sqrt((a - 1) ** 2 + 4 * y2)
    t = (a - 1 + b) / 2
    u = torch.acos(x / torch.sqrt(1 + t))
    v = torch.asinh(-torch.sign(y) * torch.sqrt(t))

    return ComplexTensor(u.to(out_dt), v.to(out_dt))


@register_complex(aten.asin)
def asin_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)

    out_dt, (x, y) = promote_real_cpu_tensors(x, y)
    y2 = y**2
    a = (x**2) + y2
    b = torch.sqrt((a - 1) ** 2 + 4 * y2)
    t = (a - 1 + b) / 2

    u = torch.arcsin(x / torch.sqrt(1 + t))
    v = torch.arcsinh(torch.sign(y) * torch.sqrt(t))

    return ComplexTensor(u.to(out_dt), v.to(out_dt))


@register_complex(aten.clone)
def clone_impl(self: ComplexTensor, *args, **kwargs) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    return ComplexTensor(torch.clone(x, *args, *kwargs), torch.clone(y, *args, **kwargs))


@register_complex(aten.cos)
def cos_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_real_cpu_tensors(x, y)
    u = torch.cos(x) * torch.cosh(y)
    v = -torch.sin(x) * torch.sinh(y)
    return ComplexTensor(u.to(out_dt), v.to(out_dt))


@register_complex(aten.cosh)
def cosh_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_real_cpu_tensors(x, y)
    u = torch.cosh(x) * torch.cos(y)
    v = torch.sinh(x) * torch.sin(y)
    return ComplexTensor(u.to(out_dt), v.to(out_dt))


@register_complex(aten.exp)
def exp_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_real_cpu_tensors(x, y)
    ex = torch.exp(x)
    u = ex * torch.cos(y)
    v = ex * torch.sin(y)
    return ComplexTensor(u.to(out_dt), v.to(out_dt))


@register_complex(aten.expm1)
def expm1_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_real_cpu_tensors(x, y)
    # TODO (hameerabbasi): The two lines below may have numerical issues
    ex = torch.exp(x)
    u = ex * torch.cos(y) - 1
    v = ex * torch.sin(y)
    return ComplexTensor(u.to(out_dt), v.to(out_dt))


@register_complex(aten.any)
def any_impl(self: ComplexTensor, *args, **kwargs) -> torch.Tensor:
    x, y = split_complex_tensor(self)
    return torch.any(x, *args, **kwargs) | torch.any(y, *args, **kwargs)


@register_complex(aten.all)
def all_impl(self: ComplexTensor, *args, **kwargs) -> torch.Tensor:
    x, y = split_complex_tensor(self)
    return torch.any(x, *args, **kwargs) & torch.any(y, *args, **kwargs)


@register_complex(aten.eq)
def eq_impl(self: ComplexTensor, rhs: ComplexTensor, *args, **kwargs) -> torch.Tensor:
    a_r, a_i = split_complex_tensor(self)
    b_r, b_i = split_complex_arg(rhs)
    return torch.eq(a_r, b_r, *args, **kwargs) & torch.eq(a_i, b_i, *args, **kwargs)


@register_complex(aten.ne)
def ne_impl(self: ComplexTensor, rhs: ComplexTensor, *args, **kwargs) -> torch.Tensor:
    a_r, a_i = split_complex_tensor(self)
    b_r, b_i = split_complex_arg(rhs)
    return torch.ne(a_r, b_r, *args, **kwargs) | torch.ne(a_i, b_i, *args, **kwargs)


@register_complex(aten.isnan)
def isnan_impl(self: ComplexTensor) -> torch.Tensor:
    re, im = split_complex_tensor(self)
    return torch.isnan(re) | torch.isnan(im)


@register_complex(aten.isclose)
def isclose_impl(
    self: ComplexTensor, rhs: ComplexTensor, rtol=1e-5, atol=1e-8, equal_nan: bool = False
) -> torch.Tensor:
    abs_diff = torch.abs(self - rhs)
    abs_other = torch.abs(rhs)
    basic_condition = abs_diff <= (rtol * abs_other + atol)

    # This is the nontrivial part
    if equal_nan:
        a_r, a_i = split_complex_tensor(self)
        b_r, b_i = split_complex_arg(rhs)

        # This logical expression makes sure that the isnan of both the real and imaginary parts
        # matches (so 1 + nan*i doesn't equal nan + 1*i)
        equal_nan_condition = (torch.isnan(a_r) == torch.isnan(b_r)) & (
            torch.isnan(a_i) == torch.isnan(b_i)
        )
        return basic_condition & equal_nan_condition

    return basic_condition


ORDERED_OPS_LIST = [
    aten.lt,
    aten.le,
    aten.gt,
    aten.ge,
    aten.maximum,
    aten.minimum,
    aten.amin,
    aten.amax,
    aten.argmin,
    aten.argmax,
    aten.clamp,
    aten.sort,
    aten.topk,
    aten.ceil,
    aten.floor,
]


def register_ordered(op: OpType):
    def ordered_impl(*args, **kwargs):
        raise NotImplementedError(
            f"{str(op).split('.', 1)!r} not implemented for {ComplexTensor.__name__!r}."
        )

    return register_complex(op, ordered_impl)


for ordered_op in ORDERED_OPS_LIST:
    globals()[f"{str(ordered_op).split('.', 1)!r}_impl"] = register_ordered(ordered_op)

del ordered_op
