from __future__ import annotations

from collections.abc import Sequence

import torch

from ..complex_tensor import ComplexTensor
from ._common import (
    COMPLEX_TO_REAL,
    ERROR_TYPES,
    OpType,
    complex_to_real_dtype,
    promote_real_cpu_tensors,
    register_binary_nonlinear,
    register_complex,
    register_error,
    register_force_test,
    register_simple,
    split_complex_arg,
    split_complex_tensor,
)

aten = torch.ops.aten


def register_binary_linear(op: OpType):
    def impl_with_alpha(
        lhs: ComplexTensor, rhs: ComplexTensor, *args, alpha, **kwargs
    ) -> ComplexTensor:
        return op(lhs, aten.mul(rhs, alpha, *args, **kwargs), *args, **kwargs)

    def impl(lhs: ComplexTensor, rhs: ComplexTensor, *args, **kwargs) -> ComplexTensor:
        alpha = kwargs.pop("alpha", None)
        if alpha is not None:
            return impl_with_alpha(lhs, rhs, *args, alpha=alpha, **kwargs)
        a_r, a_i = split_complex_arg(lhs)
        b_r, b_i = split_complex_arg(rhs)
        out_dt, (a_r, a_i, b_r, b_i) = promote_real_cpu_tensors(a_r, a_i, b_r, b_i)
        u = op(a_r, b_r, *args, **kwargs)
        v = op(a_i, b_i, *args, **kwargs)
        return ComplexTensor(u.to(out_dt), v.to(out_dt))

    return register_complex(op, impl)


# Not sure why torch dispatch does not hit here.
@register_complex(aten.real)
def real_impl(self: ComplexTensor) -> torch.Tensor:
    re, _ = split_complex_tensor(self)
    return re


# Not sure why torch dispatch does not hit here.
@register_complex(aten.imag)
def imag_impl(self: ComplexTensor) -> torch.Tensor:
    _, im = split_complex_tensor(self)
    return im


@register_complex(aten.is_pinned)
def is_pinned_impl(self: ComplexTensor, device: torch.device | None = None) -> bool:
    return self.is_pinned(device)


slice_impl = register_simple(aten.slice)
flatten_impl = register_simple(aten.flatten)
view_impl = register_simple(aten.view)
diagonal_impl = register_simple(aten.diagonal)
expand_impl = register_simple(aten.expand)
unsqueeze_impl = register_simple(aten.unsqueeze)
mean_impl = register_simple(aten.mean)
sum_impl = register_simple(aten.sum)
clone_impl = register_simple(aten.clone)
neg_impl = register_simple(aten.neg)
flip_impl = register_simple(aten.flip)
permute_impl = register_simple(aten.permute)
repeat_impl = register_simple(aten.repeat)
index_select_impl = register_simple(aten.index_select)
split_with_sizes_impl = register_simple(aten.split_with_sizes)
cumsum_impl = register_simple(aten.cumsum)
detach_impl = register_simple(aten.detach)
select_impl = register_simple(aten.select)
squeeze_impl = register_simple(aten.squeeze)
zero__impl = register_simple(aten.zero_)
transpose_impl = register_simple(aten.transpose)

# TODO (hameerabbasi): Not being tested
copy_impl = register_force_test(aten.copy, register_simple(aten.copy))
# TODO (hameerabbasi): Not being tested
_to_copy_impl = register_force_test(aten._to_copy, register_simple(aten._to_copy))
# TODO (hameerabbasi): Not being tested
col2im_impl = register_force_test(aten.col2im, register_simple(aten.col2im))
# TODO (hameerabbasi): Not being tested
alias_impl = register_force_test(aten.alias, register_simple(aten.alias))

# some binary ops which we can stamp out
mul_impl = register_binary_nonlinear(aten.mul)
mm_impl = register_binary_nonlinear(aten.mm)
dot_impl = register_binary_nonlinear(aten.dot)
bmm_impl = register_binary_nonlinear(aten.bmm)

# TODO (hameerabbasi): Not being tested
convolution_impl = register_force_test(
    aten.convolution, register_binary_nonlinear(aten.convolution)
)

add_impl = register_binary_linear(aten.add)
sub_impl = register_binary_linear(aten.sub)


@register_complex(aten.div)
def div_impl(lhs: ComplexTensor, rhs: ComplexTensor, *, rounding_mode=None):
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


@register_complex(aten.reciprocal)
def reciprocal_impl(self: ComplexTensor):
    self_r, self_i = split_complex_tensor(self)
    out_dt, (self_r, self_i) = promote_real_cpu_tensors(self_r, self_i)
    den = self_r * self_r + self_i * self_i
    return ComplexTensor(
        aten.div(self_r, den).to(out_dt),
        aten.div(-self_i, den).to(out_dt),
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


@register_complex(aten.isinf)
def isinf_impl(self: ComplexTensor) -> torch.Tensor:
    re, im = split_complex_tensor(self)
    return torch.isinf(re) | torch.isinf(im)


@register_complex(aten.isfinite)
def isfinite_impl(self: ComplexTensor) -> torch.Tensor:
    re, im = split_complex_tensor(self)
    return torch.isfinite(re) & torch.isfinite(im)


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


ERROR_OPS_LIST = [
    aten.lt,
    aten.le,
    aten.gt,
    aten.ge,
    aten.amin,
    aten.amax,
    aten.clamp,
    aten.ceil,
    aten.floor,
    aten.minimum,
    aten.maximum,
    aten.trunc,
    aten.sign,
    aten.argmax,
    aten.argmin,
    aten.sort,
    aten.topk,
    aten.round,
]


ERROR_TYPES.update(
    {
        aten.minimum: RuntimeError,
        aten.maximum: RuntimeError,
        aten.argmax: RuntimeError,
        aten.argmin: RuntimeError,
        aten.sort: RuntimeError,
        aten.topk: RuntimeError,
    }
)


for err_op in ERROR_OPS_LIST:
    globals()[f"{str(err_op).split('.', 1)}_impl"] = register_error(err_op)

del err_op


@register_complex(aten.masked_scatter)
def masked_scatter_impl(
    self: ComplexTensor, mask: torch.Tensor, source: ComplexTensor
) -> ComplexTensor:
    self_r, self_i = split_complex_tensor(self)
    source_r, source_i = split_complex_arg(source)
    ret_r = torch.masked_scatter(self_r, mask, source_r)
    ret_i = torch.masked_scatter(self_i, mask, source_i)

    return ComplexTensor(ret_r, ret_i)


@register_force_test(aten.slice_scatter)
def slice_scatter_impl(
    self: ComplexTensor,
    source: ComplexTensor,
    dim: int = 0,
    start: int | None = None,
    end: int | None = None,
    step: int = 1,
) -> ComplexTensor:
    self_r, self_i = split_complex_tensor(self)
    source_r, source_i = split_complex_arg(source)
    ret_r = torch.slice_scatter(self_r, source_r, dim=dim, start=start, end=end, step=step)
    ret_i = torch.slice_scatter(self_i, source_i, dim=dim, start=start, end=end, step=step)

    return ComplexTensor(ret_r, ret_i)


@register_complex(aten.index_put)
def index_put_impl(
    self: ComplexTensor,
    indices: tuple[torch.Tensor, ...],
    values: ComplexTensor,
    accumulate: bool = False,
) -> ComplexTensor:
    self_r, self_i = split_complex_tensor(self)
    values_r, values_i = split_complex_arg(values)
    ret_r = torch.index_put(self_r, indices, values_r, accumulate=accumulate)
    ret_i = torch.index_put(self_i, indices, values_i, accumulate=accumulate)

    return ComplexTensor(ret_r, ret_i)


@register_complex(aten.where)
def where_impl(mask: torch.Tensor, x: ComplexTensor, y: ComplexTensor) -> ComplexTensor:
    x_r, x_i = split_complex_arg(x)
    y_r, y_i = split_complex_arg(y)

    ret_r = torch.where(mask, x_r, y_r)
    ret_i = torch.where(mask, x_i, y_i)

    return ComplexTensor(ret_r, ret_i)


@register_complex(aten.full_like)
def full_like_impl(input: ComplexTensor, fill_value: complex, *args, **kwargs) -> ComplexTensor:
    input_r, input_i = split_complex_tensor(input)
    fv_r, fv_i = split_complex_arg(fill_value)

    ret_r = torch.full_like(input_r, fv_r, *args, **kwargs)
    ret_i = torch.full_like(input_i, fv_i, *args, **kwargs)

    return ComplexTensor(ret_r, ret_i)


@register_complex(aten.cat)
def cat_impl(tensors: Sequence[ComplexTensor], dim: int = 0) -> ComplexTensor:
    tensors_r = []
    tensors_i = []

    for t in tensors:
        t_r, t_i = split_complex_arg(t)
        tensors_r.append(t_r)
        tensors_i.append(t_i)

    ret_r = torch.cat(tensors_r, dim=dim)
    ret_i = torch.cat(tensors_i, dim=dim)

    return ComplexTensor(ret_r, ret_i)


@register_complex(aten.sgn)
def sgn_impl(self: ComplexTensor) -> ComplexTensor:
    self_r, self_i = split_complex_tensor(self)
    out_dt, (self_r, self_i) = promote_real_cpu_tensors(self_r, self_i)
    mask = self != 0
    abs_self = torch.abs(ComplexTensor(self_r, self_i))
    masked_sgn = ComplexTensor(
        torch.div(self_r, abs_self).to(out_dt), torch.div(self_i, abs_self).to(out_dt)
    )
    return torch.where(mask, masked_sgn, 0)


@register_complex(aten.sqrt)
def sqrt_impl(self: ComplexTensor) -> ComplexTensor:
    self_r, self_i = split_complex_tensor(self)
    out_dt, (self_r, self_i) = promote_real_cpu_tensors(self_r, self_i)
    self = ComplexTensor(self_r, self_i)
    self_abs_sqrt = torch.sqrt(torch.abs(self))
    self_half_angle = torch.angle(self) / 2

    ret_r = self_abs_sqrt * torch.cos(self_half_angle)
    ret_i = self_abs_sqrt * torch.sin(self_half_angle)

    return ComplexTensor(ret_r.to(out_dt), ret_i.to(out_dt))


@register_complex(aten.rsqrt)
def rsqrt_impl(self: ComplexTensor) -> ComplexTensor:
    self_r, self_i = split_complex_tensor(self)
    out_dt, (self_r, self_i) = promote_real_cpu_tensors(self_r, self_i)
    self = ComplexTensor(self_r, self_i)
    self_abs_rsqrt = torch.rsqrt(torch.abs(self))
    self_neg_half_angle = -torch.angle(self) / 2

    ret_r = self_abs_rsqrt * torch.cos(self_neg_half_angle)
    ret_i = self_abs_rsqrt * torch.sin(self_neg_half_angle)

    return ComplexTensor(ret_r.to(out_dt), ret_i.to(out_dt))


@register_complex(aten.addmm)
def addmm_impl(
    input: ComplexTensor,
    mat1: ComplexTensor,
    mat2: ComplexTensor,
    out_dtype: torch.dtype | None = None,
    beta: complex = 1,
    alpha: complex = 1,
) -> ComplexTensor:
    ret = alpha * input + beta * torch.mm(mat1, mat2)
    ret_r, ret_i = split_complex_tensor(ret)
    if out_dtype is not None:
        out_dtype = COMPLEX_TO_REAL[out_dtype]
        ret_r, ret_i = ret_r.to(out_dtype), ret_i.to(out_dtype)
    return ComplexTensor(ret_r, ret_i)


def elemwise_nonzero(self: ComplexTensor) -> torch.Tensor:
    re, im = split_complex_tensor(self)
    return (re != 0) | (im != 0)


def register_nonzero_impl(op: OpType):
    def nonzero_impl(self: ComplexTensor, other: ComplexTensor, *args, **kwargs) -> torch.Tensor:
        return op(elemwise_nonzero(self), elemwise_nonzero(other), *args, **kwargs)

    func_name = f"{str(op).split('.', 1)}_impl"
    nonzero_impl.__name__ = func_name
    nonzero_impl.__qualname__ = func_name

    return register_complex(op, nonzero_impl)


logical_and_impl = register_nonzero_impl(aten.logical_and)
logical_or_impl = register_nonzero_impl(aten.logical_or)
logical_xor_impl = register_nonzero_impl(aten.logical_xor)


@register_complex(aten.logical_not)
def logical_not_impl(self: ComplexTensor, *args, **kwargs) -> torch.Tensor:
    return torch.logical_not(elemwise_nonzero(self), *args, **kwargs)


@register_complex(aten.view_as_real)
def view_as_real_impl(self: ComplexTensor) -> torch.Tensor:
    re, im = split_complex_tensor(self)
    return torch.stack([re, im], dim=-1)


@register_complex(aten.linalg_vector_norm)
def linalg_vector_norm_impl(self: ComplexTensor, *args, **kwargs) -> torch.Tensor:
    return torch.linalg.vector_norm(torch.abs(self), *args, **kwargs)


@register_force_test(aten.copy_)
def copy__impl(self: ComplexTensor, src, *args, **kwargs):
    self_re, self_im = split_complex_tensor(self)
    src_re, src_im = split_complex_arg(src)

    ret_re = self_re.copy_(src_re, *args, **kwargs)
    ret_im = self_im.copy_(src_im, *args, **kwargs)

    return ComplexTensor(ret_re, ret_im)


@register_complex(aten.new_zeros)
def new_zeros_impl(
    self: ComplexTensor, size, *, dtype=None, **kwargs
) -> ComplexTensor | torch.Tensor:
    self_re, self_im = split_complex_tensor(self)
    if dtype is not None and dtype not in COMPLEX_TO_REAL:
        return self_re.new_zeros(size, dtype=dtype, **kwargs)

    if dtype is not None:
        dtype = COMPLEX_TO_REAL[dtype]
    re = self_re.new_zeros(size, dtype=dtype, **kwargs)
    im = self_im.new_zeros(size, dtype=dtype, **kwargs)

    return ComplexTensor(re, im)


@register_complex(aten._local_scalar_dense)
def _local_scalar_dense_impl(self: ComplexTensor, *args, **kwargs) -> complex:
    x, y = split_complex_tensor(self)
    u = aten._local_scalar_dense(x, *args, **kwargs)
    v = aten._local_scalar_dense(y, *args, **kwargs)
    return complex(u, v)


@register_complex(aten.allclose)
def allclose_impl(
    input: torch.Tensor,
    other: torch.Tensor,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> torch.Tensor:
    return torch.all(torch.isclose(input, other, rtol=rtol, atol=atol, equal_nan=equal_nan))


@register_complex(aten.stack)
def stack_impl(self: list[ComplexTensor], *args, **kwargs) -> ComplexTensor:
    re_im_tuples = [split_complex_arg(self_i) for self_i in self]
    u = torch.stack([c[0] for c in re_im_tuples], *args, **kwargs)
    v = torch.stack([c[1] for c in re_im_tuples], *args, **kwargs)
    return ComplexTensor(u, v)


@register_complex(aten.randn_like)
def randn_like_impl(self: ComplexTensor, *, dtype=None, **kwargs) -> ComplexTensor | torch.Tensor:
    if dtype is not None and dtype not in COMPLEX_TO_REAL:
        return torch.randn_like(self.re, dtype=dtype, **kwargs)

    if dtype is not None:
        dtype = COMPLEX_TO_REAL[dtype]

    self_re, self_im = split_complex_tensor(self)
    ret_re = torch.randn_like(self_re, dtype=dtype, **kwargs) / 2
    ret_im = torch.randn_like(self_im, dtype=dtype, **kwargs) / 2
    return ComplexTensor(ret_re, ret_im)
