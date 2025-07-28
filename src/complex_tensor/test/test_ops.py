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

COMPLEX_SKIP = {
    "real": "Does not hit __torch_dispatch__",
    "imag": "Does not hit __torch_dispatch__",
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
        if op.name in COMPLEX_SKIP:
            self.skipTest(COMPLEX_SKIP[op.name])

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
