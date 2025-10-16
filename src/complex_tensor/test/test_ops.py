from __future__ import annotations

import torch
from torch._ops import OpOverload
from torch.testing._internal.common_device_type import OpDTypes, instantiate_device_type_tests, ops
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import (
    TestGradients,
    parametrize,
    run_tests,
    unMarkDynamoStrictTest,
)
from torch.testing._internal.opinfo.core import OpInfo

from complex_tensor.ops import COMPLEX_OPS_TABLE, FORCE_TEST_LIST
from complex_tensor.ops._common import ComplexDispatchMode, _as_complex_tensor
from complex_tensor.test.utils import (
    COMPLEX_DTYPES,
    TestCase,
    TestDescriptor,
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
    return aten_op._qualified_op_name.split("::")[-1]


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
    TestDescriptor(op_name="empty_like"): "Inconsistent output",
    # This passes with `PYTORCH_OPINFO_SAMPLE_INPUT_INDEX=35 ...
    # but when the whole test is run, it fails with this exact
    # sample.
    TestDescriptor(op_name="repeat", compile=True): "Heisenbug",
    TestDescriptor(
        op_name="allclose", compile=True
    ): "`aten.allclose` requires data-dependent control-flow",
    TestDescriptor(op_name="randn_like"): "Inconsistent output",
}

EXTRA_KWARGS = {
    TestDescriptor(op_name="asinh", dtype=torch.complex64, gradcheck=False): {
        "rtol": 2e-5,
        "atol": 5e-5,
    },
    TestDescriptor(op_name="tanh", dtype=torch.complex64, gradcheck=False): {
        "rtol": 1e-4,
        "atol": 1e-5,
    },
    TestDescriptor(op_name="pow", dtype=torch.complex64, gradcheck=False): {
        "rtol": 2e-2,
        "atol": 2e-6,
    },
}


class TestComplexTensor(TestCase):
    _default_dtype_check_enabled = True

    @parametrize("compile", [False, True])
    @ops(implemented_op_db, dtypes=OpDTypes.supported, allowed_dtypes=list(COMPLEX_DTYPES))
    def test_consistency(self, device, dtype, op: OpInfo, compile: bool):
        self.check_consistency(device, dtype, op, compile)

    @parametrize("compile", [False, True])
    @ops(force_test_op_db, allowed_dtypes=list(COMPLEX_DTYPES))
    def test_maybe_error(self, device, dtype, op: OpInfo, compile: bool):
        self.check_consistency(device, dtype, op, compile)

    def check_consistency(self, device: torch.device, dtype, op: OpInfo, compile: bool) -> None:
        test_info = TestDescriptor(
            op_name=op.name, device=device, dtype=dtype, compile=compile, gradcheck=False
        )
        for xfail_info, reason in SKIPS.items():
            if xfail_info.matches(test_info):
                self.skipTest(reason)

        kwargs = {}
        for extra_info, extra_kw in EXTRA_KWARGS.items():
            if extra_info.matches(test_info):
                kwargs = extra_kw
                break

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

            self.assertSameResult(expected, actual, ignore_exc_types=compile, **kwargs)


@unMarkDynamoStrictTest
class TestComplexBwdGradients(TestGradients):
    @ops(implemented_op_db, dtypes=OpDTypes.supported_backward, allowed_dtypes=[torch.complex128])
    def test_fn_grad(self, device: torch.device, dtype: torch.dtype, op: OpInfo) -> None:
        test_info = TestDescriptor(
            op_name=op.name, device=device, dtype=dtype, compile=False, gradcheck=True
        )
        for xfail_info, reason in SKIPS.items():
            if xfail_info.matches(test_info):
                self.skipTest(reason)

        if dtype not in op.supported_backward_dtypes(torch.device(device).type):
            self.skipTest(f"Skipped! {dtype=} is not in supported backward dtypes!")

        with ComplexDispatchMode():
            self._grad_test_helper(device, dtype, op, op.get_op())


instantiate_device_type_tests(TestComplexTensor, globals())
instantiate_device_type_tests(TestComplexBwdGradients, globals())

if __name__ == "__main__":
    run_tests()
