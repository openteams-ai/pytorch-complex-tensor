from __future__ import annotations

import torch
from torch._ops import OpOverload
from torch.testing._internal.common_device_type import instantiate_device_type_tests, ops
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import TestCase, parametrize, run_tests
from torch.testing._internal.opinfo.core import OpInfo

from complex_tensor import ComplexTensor
from complex_tensor.ops.core import COMPLEX_OPS_TABLE, ORDERED_OPS_LIST

from .utils import TestDescriptor

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


ordered_op_names = set(map(_get_opname_from_aten_op, ORDERED_OPS_LIST))
implemented_op_names = (
    set(map(_get_opname_from_aten_op, COMPLEX_OPS_TABLE.keys())) - ordered_op_names
)
implemented_op_db = tuple(filter(lambda op: op.name in implemented_op_names, complex_op_db))
ordered_op_db = tuple(filter(lambda op: op.name in ordered_op_names, op_db))


SKIPS = {
    TestDescriptor(op_name="real"): "`aten.real` does not hit `__torch_dispatch__`",
    TestDescriptor(op_name="imag"): "`aten.imag` does not hit `__torch_dispatch__`",
    TestDescriptor(
        op_name="sub", dtype=torch.complex32, compile=True
    ): "numerical precision optimized out",
    TestDescriptor(
        op_name="sort"
    ): "https://github.com/pytorch/pytorch/pull/159556#issuecomment-3154215299",
    TestDescriptor(
        op_name="minimum"
    ): "https://github.com/pytorch/pytorch/pull/159556#issuecomment-3154215299",
    TestDescriptor(
        op_name="maximum"
    ): "https://github.com/pytorch/pytorch/pull/159556#issuecomment-3154215299",
    TestDescriptor(
        op_name="argmin"
    ): "https://github.com/pytorch/pytorch/pull/159556#issuecomment-3154215299",
    TestDescriptor(
        op_name="argmax"
    ): "https://github.com/pytorch/pytorch/pull/159556#issuecomment-3154215299",
    TestDescriptor(
        op_name="topk"
    ): "https://github.com/pytorch/pytorch/pull/159556#issuecomment-3154215299",
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
    @ops(implemented_op_db, allowed_dtypes=list(complex_types))
    def test_consistency(self, device, dtype, op: OpInfo, compile: bool):
        test_info = TestDescriptor(op_name=op.name, device=device, dtype=dtype, compile=compile)
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

    @ops(ordered_op_db, dtypes=list(complex_types))
    def test_ordered_raises(self, device, dtype, op: OpInfo):
        test_info = TestDescriptor(op_name=op.name, device=device, dtype=dtype)
        for xfail_info, reason in SKIPS.items():
            if xfail_info.matches(test_info):
                self.skipTest(reason)

        sample_inputs = op.sample_inputs(device, dtype)

        for sample_input in sample_inputs:
            subclass_sample = sample_input.transform(_as_complex_tensor)
            self.assertRaises(
                NotImplementedError,
                op,
                sample_input.input,
                *sample_input.args,
                **sample_input.kwargs,
            )
            self.assertRaises(
                NotImplementedError,
                op,
                subclass_sample.input,
                *subclass_sample.args,
                **subclass_sample.kwargs,
            )


instantiate_device_type_tests(TestComplexTensor, globals())

if __name__ == "__main__":
    run_tests()
