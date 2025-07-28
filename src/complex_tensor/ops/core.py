from typing import Callable, Optional, Union

import torch
from torch._ops import OpOverload
from torch._refs import is_complex

from complex_tensor import ComplexTensor

TableType = dict[OpOverload, Callable]
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
    ops: Union[list[OpOverload], OpOverload],
    func_impl: Optional[Callable] = None,
):
    """Decorator to register an implementation for some ops in some dispatch tables"""
    if not isinstance(ops, list):
        ops = [ops]

    def inner(func):
        for op in ops:
            COMPLEX_OPS_TABLE[op] = func
        return func

    if func_impl is None:
        return inner
    return inner(func_impl)


def lookup_complex(func, *args, **kwargs):
    return COMPLEX_OPS_TABLE.get(func, COMPLEX_OPS_TABLE.get(func.overloadpacket, None))


def split_complex_arg(arg) -> tuple[torch.Tensor, torch.Tensor]:
    # todo(amjames): Not really handling all cases we assume bare tensors are
    # real data types and we assume scalar args will be python float/complex
    # only.
    if isinstance(arg, ComplexTensor):
        return split_complex_tensor(arg)
    if isinstance(arg, torch.Tensor):
        if is_complex(arg):
            return arg.real, arg.imag
        return arg, torch.zeros_like(arg)
    if isinstance(arg, complex):
        return arg.real, arg.complex
    if isinstance(arg, (float, int, bool)):
        return arg, type(arg)(False)
    raise TypeError(f"Expected tensor or number got, {type(arg)}")


def split_complex_tensor(complex_tensor: ComplexTensor) -> tuple[torch.Tensor, torch.Tensor]:
    return complex_tensor.re, complex_tensor.im


def complex_to_real_dtype(dtype: torch.dtype) -> torch.dtype:
    return COMPLEX_TO_REAL.get(dtype, dtype)


# Not sure why torch dispatch does not hit here.
@register_complex(aten.real)
def real(self):
    re, _ = split_complex_tensor(self)
    return re


# Not sure why torch dispatch does not hit here.
@register_complex(aten.imag)
def imag(self):
    _, im = split_complex_tensor(self)
    return im


def promote_real_cpu_tensors(
    tensor: torch.Tensor, *tensors: torch.Tensor
) -> tuple[torch.dtype, tuple[torch.Tensor, ...]]:
    out_dt = tensor.dtype
    for t in tensors:
        out_dt = torch.promote_types(out_dt, t.dtype)

    prom_dt = PROMOTE_TYPES_CPU.get(out_dt)
    if (
        out_dt is None
        or tensor.device.type != "cpu"
        or any(t.device.type != "cpu" for t in tensors)
    ):
        return out_dt, (tensor, *tensors)

    return out_dt, (tensor.to(prom_dt), *(t.to(prom_dt) for t in tensors))


def register_binary_nonlinear(aten_op):
    def impl(lhs: ComplexTensor, rhs: ComplexTensor, *args, **kwargs) -> ComplexTensor:
        a_r, a_i = split_complex_tensor(lhs)
        b_r, b_i = split_complex_arg(rhs)
        out_dt, (a_r, a_i, b_r, b_i) = promote_real_cpu_tensors(a_r, a_i, b_r, b_i)
        real = aten_op(a_r, b_r, *args, **kwargs) - aten_op(a_i, b_i, *args, **kwargs)
        imag = aten_op(a_r, b_i, *args, **kwargs) + aten_op(a_i, b_r, *args, **kwargs)
        return ComplexTensor(real.to(out_dt), imag.to(out_dt))

    return register_complex(aten_op, impl)


def register_binary_linear(aten_op):
    def impl(lhs: ComplexTensor, rhs: ComplexTensor, *args, **kwargs) -> ComplexTensor:
        a_r, a_i = split_complex_tensor(lhs)
        b_r, b_i = split_complex_arg(rhs)
        r = aten_op(a_r, b_r, *args, **kwargs)
        i = aten_op(a_i, b_i, *args, **kwargs)
        return ComplexTensor(r, i)

    return register_complex(aten_op, impl)


def _make_simple(aten_op):
    def impl(self: ComplexTensor, *args, **kwargs) -> ComplexTensor:
        x, y = split_complex_tensor(self)
        u = aten_op(x, *args, **kwargs)
        v = aten_op(y, *args, **kwargs)
        return ComplexTensor(u, v)

    return impl


def register_simple(aten_op):
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
def prod_impl(self, *args, **kwargs):
    dtype = kwargs.pop("dtype", self.dtype)
    kwargs["dtype"] = complex_to_real_dtype(dtype)

    prod_r = torch.prod(torch.abs(self), *args, **kwargs)
    sum_phi = torch.sum(torch.angle(self), *args, **kwargs)
    u = prod_r * torch.cos(sum_phi)
    v = prod_r * torch.sin(sum_phi)
    return ComplexTensor(u, v)


@register_complex(aten.sum)
def sum_impl(self, *args, **kwargs):
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
def acos_impl(self: ComplexTensor):
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
def asin_impl(self: ComplexTensor):
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
def clone_impl(self: ComplexTensor, *args, **kwargs):
    x, y = split_complex_tensor(self)
    return ComplexTensor(torch.clone(x, *args, *kwargs), torch.clone(y, *args, **kwargs))


@register_complex(aten.cos)
def cos_impl(self: ComplexTensor):
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_real_cpu_tensors(x, y)
    u = torch.cos(x) * torch.cosh(y)
    v = -torch.sin(x) * torch.sinh(y)
    return ComplexTensor(u.to(out_dt), v.to(out_dt))


@register_complex(aten.cosh)
def cosh_impl(self: ComplexTensor):
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_real_cpu_tensors(x, y)
    u = torch.cosh(x) * torch.cos(y)
    v = torch.sinh(x) * torch.sin(y)
    return ComplexTensor(u.to(out_dt), v.to(out_dt))


@register_complex(aten.exp)
def exp_impl(self: ComplexTensor):
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_real_cpu_tensors(x, y)
    ex = torch.exp(x)
    u = ex * torch.cos(y)
    v = ex * torch.sin(y)
    return ComplexTensor(u.to(out_dt), v.to(out_dt))


@register_complex(aten.expm1)
def expm1_impl(self: ComplexTensor):
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_real_cpu_tensors(x, y)
    # TODO (hameerabbasi): The two lines below may have numerical issues
    ex = torch.exp(x)
    u = ex * torch.cos(y) - 1
    v = ex * torch.sin(y)
    return ComplexTensor(u.to(out_dt), v.to(out_dt))


@register_complex(aten.any)
def any_impl(self, *args, **kwargs):
    x, y = split_complex_tensor(self)
    return torch.logical_or(torch.any(x, *args, **kwargs), torch.any(y, *args, **kwargs))


@register_complex(aten.all)
def all_impl(self, *args, **kwargs):
    x, y = split_complex_tensor(self)
    return torch.logical_and(torch.any(x, *args, **kwargs), torch.any(y, *args, **kwargs))
