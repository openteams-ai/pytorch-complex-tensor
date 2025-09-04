from __future__ import annotations

import torch
from torch._ops import OpOverload
from torch.testing._internal.common_device_type import instantiate_device_type_tests, ops
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import (
    TestGradients,
    parametrize,
    run_tests,
    unMarkDynamoStrictTest,
)
from torch.testing._internal.opinfo.core import OpInfo

from complex_tensor.ops import COMPLEX_OPS_TABLE, FORCE_TEST_LIST
from complex_tensor.ops._common import ComplexDispatchMode
from complex_tensor.test.utils import (
    COMPLEX_DTYPES,
    TestCase,
    TestDescriptor,
    _as_complex_tensor,
)

torch._dynamo.config.recompile_limit = float("inf")
torch._dynamo.config.accumulated_recompile_limit = float("inf")

aten = torch.ops.aten

complex_op_db = tuple(
    filter(lambda op: any(op.supports_dtype(ct, "cpu") for ct in COMPLEX_DTYPES), op_db)
)


def _get_opname_from_aten_op(aten_op):
    if isinstance(aten_op, OpOverload):
        aten_op = aten_op.overloadpacket
    _, name = str(aten_op).split(".", 1)
    return name


force_test_names = set(map(_get_opname_from_aten_op, FORCE_TEST_LIST))
implemented_op_names = (
    set(map(_get_opname_from_aten_op, COMPLEX_OPS_TABLE.keys())) - force_test_names
)
implemented_op_db = tuple(filter(lambda op: op.name in implemented_op_names, complex_op_db))
force_test_op_db = tuple(filter(lambda op: op.name in force_test_names, op_db))

tested_op_names = {op.name for op in implemented_op_db} | {op.name for op in force_test_op_db}
non_tested_ops = {
    op for op in COMPLEX_OPS_TABLE if _get_opname_from_aten_op(op) not in tested_op_names
}

if len(non_tested_ops) != 0:
    import textwrap
    import warnings

    list_missing_ops = "\n".join(
        sorted([op._qualified_op_name.replace("::", ".") for op in non_tested_ops])
    )
    warnings.warn(
        "Not all implemented ops are tested. List of ops missing tests:"
        f"\n{textwrap.indent(list_missing_ops, '    ')}",
        UserWarning,
        stacklevel=2,
    )


SKIPS = {
    TestDescriptor(op_name="real"): "`aten.real` does not hit `__torch_dispatch__`",
    TestDescriptor(op_name="imag"): "`aten.imag` does not hit `__torch_dispatch__`",
    TestDescriptor(op_name="repeat", dtype=torch.complex64, compile=True): "Heisenbug",
    TestDescriptor(op_name="repeat", dtype=torch.complex128, compile=True): "Heisenbug",
    TestDescriptor(
        op_name="allclose", compile=True
    ): "`aten.allclose` requires data-dependent control-flow",
    TestDescriptor(
        op_name="randn_like", compile=True
    ): "`aten.randn_like` doesn't support `torch.compile`",
}


class TestComplexTensor(TestCase):
    _default_dtype_check_enabled = True

    @parametrize("compile", [False, True])
    @ops(implemented_op_db, allowed_dtypes=list(COMPLEX_DTYPES))
    def test_consistency(self, device, dtype, op: OpInfo, compile: bool):
        self.check_consistency(device, dtype, op, compile)

    @parametrize("compile", [False, True])
    @ops(force_test_op_db, dtypes=list(COMPLEX_DTYPES))
    def test_maybe_error(self, device, dtype, op: OpInfo, compile: bool):
        self.check_consistency(device, dtype, op, compile)

    def check_consistency(self, device, dtype, op: OpInfo, compile: bool) -> None:
        test_info = TestDescriptor(op_name=op.name, device=device, dtype=dtype, compile=compile)
        for xfail_info, reason in SKIPS.items():
            if xfail_info.matches(test_info):
                self.skipTest(reason)

        sample_inputs = op.sample_inputs(device, dtype)
        op_eager = op
        if compile:
            op = torch.compile(op, fullgraph=True)

        for sample_input in sample_inputs:

            def expected(sample_input=sample_input):
                return op_eager(sample_input.input, *sample_input.args, **sample_input.kwargs)

            subclass_sample = sample_input.transform(_as_complex_tensor)

            def actual(subclass_sample=subclass_sample):
                return op(subclass_sample.input, *subclass_sample.args, **subclass_sample.kwargs)

            self.assertSameResult(expected, actual, ignore_exc_types=compile)


@unMarkDynamoStrictTest
class TestComplexBwdGradients(TestGradients):
    @ops(implemented_op_db, allowed_dtypes=list(COMPLEX_DTYPES))
    def test_fn_grad(self, device, dtype, op: OpInfo) -> None:
        if dtype not in op.supported_backward_dtypes(torch.device(device).type):
            self.skipTest("Skipped! Dtype is not in supported backward dtypes!")

        with ComplexDispatchMode():
            op.gradcheck_fast_mode = False
            op.check_batched_grad = False
            self._grad_test_helper(device, dtype, op, op.get_op())


instantiate_device_type_tests(TestComplexTensor, globals())
instantiate_device_type_tests(TestComplexBwdGradients, globals())

if __name__ == "__main__":
    run_tests()
