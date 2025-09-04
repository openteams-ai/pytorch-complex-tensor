import torch

from ..complex_tensor import ComplexTensor
from ._common import (
    complex_to_real_dtype,
    register_complex,
    register_force_test,
    split_complex_tensor,
)

prims = torch.ops.prims


# TODO (hameerabbasi): Not being tested
@register_force_test(prims.convert_element_type)
def convert_element_type_impl(x: ComplexTensor, dtype: torch.dtype) -> ComplexTensor:
    dtype = complex_to_real_dtype(dtype)
    u, v = split_complex_tensor(x)
    u_out = prims.convert_element_type(u, dtype)
    v_out = prims.convert_element_type(v, dtype)

    return ComplexTensor(u_out, v_out)


@register_complex(prims.conj_physical)
@register_complex(prims.conj)
def conj_physical_impl(self: ComplexTensor) -> ComplexTensor:
    re, im = split_complex_tensor(self)
    return ComplexTensor(re, -im)
