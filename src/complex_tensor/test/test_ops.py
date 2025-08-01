from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import torch
from torch._ops import OpOverload
from torch.testing._internal.common_device_type import instantiate_device_type_tests, ops
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import TestCase, parametrize, run_tests
from torch.testing._internal.opinfo.core import OpInfo

from complex_tensor import ComplexTensor
from complex_tensor.ops.core import COMPLEX_OPS_TABLE

torch._dynamo.config.recompile_limit = float("inf")

complex_types = {torch.complex128, torch.complex64, torch.complex32}
aten = torch.ops.aten

complex_op_db = tuple(
    filter(lambda op: any(op.supports_dtype(ct, "cpu") for ct in complex_types), op_db)
)


def _get_opname_from_aten_op(aten_op):
    if isinstance(aten_op, OpOverload):
        aten_op = aten_op.overloadpacket
    _, name = str(aten_op).split(".", 1)
    return name


implemented_op_names = set(map(_get_opname_from_aten_op, COMPLEX_OPS_TABLE.keys()))
implemented_op_db = tuple(filter(lambda op: op.name in implemented_op_names, complex_op_db))


@dataclass(frozen=True)
class TestInfo:
    op_name: str | None = field(default=None)
    device: str | None = field(default=None)
    dtype: torch.dtype | None = field(default=None)
    compile: bool | None = field(default=None)

    def matches(self, other: TestInfo) -> bool:
        fields1 = dataclasses.fields(self)
        fields2 = dataclasses.fields(other)
        if fields1 != fields2:
            return False

        for f in fields1:
            f1 = getattr(self, f.name)
            f2 = getattr(other, f.name)
            if f1 is not None and f2 is not None and f1 != f2:
                return False

        return True


SKIPS = {
    TestInfo(op_name="real"): "`aten.real` does not hit `__torch_dispatch__`",
    TestInfo(op_name="imag"): "`aten.imag` does not hit `__torch_dispatch__`",
    TestInfo(
        op_name="sub", dtype=torch.complex32, compile=True
    ): "numerical precision optimized out",
}


def _as_complex_tensor(arg):
    if (
        not isinstance(arg, ComplexTensor)
        and isinstance(arg, torch.Tensor)
        and arg.dtype in complex_types
    ):
        return ComplexTensor.from_interleaved(arg)
    return arg


class TestComplexTensor(TestCase):
    _default_dtype_check_enabled = True

    @parametrize("compile", [False, True])
    @ops(implemented_op_db, allowed_dtypes=complex_types)
    def test_consistency(self, device, dtype, op: OpInfo, compile: bool):
        test_info = TestInfo(op_name=op.name, device=device, dtype=dtype, compile=compile)
        for xfail_info, reason in SKIPS.items():
            if xfail_info.matches(test_info):
                self.skipTest(reason)

        sample_inputs = op.sample_inputs(device, dtype)
        op_eager = op
        if compile:
            op = torch.compile(op, fullgraph=True)

        for sample_input in sample_inputs:
            interleaved_input = sample_input.input
            interleaved_args = sample_input.args
            interleaved_kwargs = sample_input.kwargs
            expected = op_eager(interleaved_input, *interleaved_args, **interleaved_kwargs)

            subclass_sample = sample_input.transform(_as_complex_tensor)
            actual = op(subclass_sample.input, *subclass_sample.args, **subclass_sample.kwargs)
            if torch.is_complex(expected):
                self.assertEqual(actual.real, expected.real)
                self.assertEqual(actual.imag, expected.imag)
            else:
                self.assertEqual(actual, expected)
                self.assertTrue(type(actual) is type(expected))


instantiate_device_type_tests(TestComplexTensor, globals())

if __name__ == "__main__":
    run_tests()
