from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import Any

import torch
from torch.testing._internal.common_utils import TestCase as PytorchTestCase
from torch.utils._pytree import tree_flatten

from complex_tensor.ops._common import COMPLEX_TO_REAL, _as_complex_tensor

COMPLEX_DTYPES = set(COMPLEX_TO_REAL)


@dataclass(frozen=True)
class TestDescriptor:
    op_name: str | None = field(default=None)
    device: str | None = field(default=None)
    dtype: torch.dtype | None = field(default=None)
    compile: bool | None = field(default=None)

    def matches(self, other: TestDescriptor) -> bool:
        fields1 = fields(self)
        fields2 = fields(other)
        if fields1 != fields2:
            return False

        for f in fields1:
            f1 = getattr(self, f.name)
            f2 = getattr(other, f.name)
            if f1 is not None and f2 is not None and f1 != f2:
                return False

        return True


class TestCase(PytorchTestCase):
    def assertSameResult(
        self,
        f1: Callable[[], Any],
        f2: Callable[[], Any],
        ignore_exc_types: bool = False,
        *args,
        **kwargs,
    ) -> None:
        try:
            result_1 = f1()
            exception_1 = None
        except Exception as e:  # noqa: BLE001
            result_1 = None
            exception_1 = e

        try:
            result_2 = f2()
            exception_2 = None
        except Exception as e:  # noqa: BLE001
            result_2 = None
            exception_2 = e
        # Special case: compiled versions don't match the error type exactly.
        if ((exception_1 is None) != (exception_2 is None)) or not ignore_exc_types:
            self.assertIs(type(exception_1), type(exception_2), f"\n{exception_1=}\n{exception_2=}")

        if exception_1 is None:
            flattened_1, spec_1 = tree_flatten(result_1)
            flattened_2, spec_2 = tree_flatten(result_2)

            self.assertEqual(
                spec_1, spec_2, "Both functions must return a result with the same tree structure."
            )
            for f1, f2 in zip(flattened_1, flattened_2, strict=False):
                f1 = _as_complex_tensor(f1)
                f2 = _as_complex_tensor(f1)

                self.assertEqual(f1, f2, *args, **kwargs)
