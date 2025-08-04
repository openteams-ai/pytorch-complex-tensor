from __future__ import annotations

import torch

from typing_extensions import Any, Self


class ComplexTensor(torch.Tensor):
    re: torch.Tensor
    im: torch.Tensor

    def __new__(cls, real: torch.Tensor, imag: torch.Tensor) -> Self:
        shape = real.shape
        device = real.device

        # TODO (hameerabbasi): `torch.compile` fails here without making these contiguous.
        # Why?
        real = real.contiguous()
        imag = imag.contiguous()

        # TODO (ajames):
        # What should we do with dtype?
        # We could convert to the complex type (float32 -> complex64), but we
        # can't use that model for say `bfloat16` which does not have a
        # corresponding complex dtype.
        # If we want to support this complex rep using any float type (see
        # https://github.com/pytorch/pytorch/issues/95100)
        # We either need to:
        # 1) add the complex types for say `complexbf32`, knowing they can't really be used anywhere
        #    else.
        # 2) We use the real float dtype here, and it is up to the user to know
        #    that dtype=float<size> here really means complex<2xSize> with dtype
        #    matching that of re/im parts alone
        # I'm going with 2 for now, so that I can test impl details with complex
        # bfloat, but might want to discuss this in the RFP
        dtype = real.dtype
        storage_offset = real.storage_offset()
        strides = real.stride()
        layout = real.layout
        requires_grad = real.requires_grad

        assert shape == imag.shape, f"Expected imag shape {shape}, got {imag.shape}"
        assert device == imag.device, f"Expected imag device {device}, got {imag.device}"
        assert dtype == imag.dtype, f"Expected imag dtype {dtype}, got {imag.dtype}"
        assert layout == imag.layout, f"Expected imag layout {layout}, got {imag.layout}"
        assert requires_grad == imag.requires_grad, (
            f"Expected imag requires_grad {requires_grad}, got {imag.requires_grad}"
        )
        res = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            shape,
            device=device,
            dtype=dtype,
            storage_offset=storage_offset,
            strides=strides,
            layout=layout,
            requires_grad=False,
        )
        res.re = real
        res.im = imag

        return res

    @property
    def real(self) -> torch.Tensor:
        return self.re

    @property
    def imag(self) -> torch.Tensor:
        return self.im

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        from .ops.core import lookup_complex

        kwargs = {} if kwargs is None else kwargs

        fn = lookup_complex(func, *args, **kwargs)
        if fn is not None:
            return fn(*args, **kwargs)

        return NotImplemented

    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def from_interleaved(t: torch.Tensor) -> ComplexTensor:
        t_real = t.real
        t_imag = t.imag if t.dtype.is_complex else torch.zeros_like(t_real)
        return ComplexTensor(t_real, t_imag)

    def as_interleaved(self) -> torch.Tensor:
        return torch.complex(self.re, self.im)

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: dict[str, torch.Tensor],
        meta: Any,
        outer_size: tuple[int, ...],
        outer_stride: tuple[int, ...],
    ) -> ComplexTensor:
        assert meta is None
        re, im = inner_tensors["re"], inner_tensors["im"]
        return ComplexTensor(re, im)

    def __tensor_flatten__(self) -> tuple[list[str], Any]:
        return ["re", "im"], None

    def __repr__(self) -> str:
        return f"ComplexTensor(real={self.re}, imag={self.im})"

    # TODO: Nested has these, but I am unsure what they are used for so that
    # will be the first step to implementing them correctly
    # __tensor_unflatten__
    # __tensor_flatten
    # __reduce_ex__


if __name__ == "__main__":

    @torch.compile()
    def f(x, y):
        return x @ y

    x = ComplexTensor(torch.tensor([[1]]), torch.tensor([[2]]))
    y = ComplexTensor(torch.tensor([[3]]), torch.tensor([[4]]))

    print(f(x, y))  # (1 + 2i) * (3 + 4i) = -5 + 10i
