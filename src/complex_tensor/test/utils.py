from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import Any

import torch
from torch.testing._internal.common_utils import TestCase as PytorchTestCase
from torch.utils._pytree import tree_flatten

from complex_tensor.ops._common import COMPLEX_TO_REAL, _as_interleaved

COMPLEX_DTYPES = set(COMPLEX_TO_REAL)


@dataclass(frozen=True)
class TestDescriptor:
    op_name: str | None = field(default=None)
    device: str | None = field(default=None)
    dtype: torch.dtype | None = field(default=None)
    compile: bool | None = field(default=None)
    gradcheck: bool | None = field(default=None)

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
        expected: Callable[[], Any],
        actual: Callable[[], Any],
        ignore_exc_types: bool = False,
        *args,
        **kwargs,
    ) -> None:
        try:
            result_e = expected()
            exception_e = None
        except Exception as e:  # noqa: BLE001
            result_e = None
            exception_e = e

        try:
            result_a = actual()
            exception_a = None
        except Exception as e:  # noqa: BLE001
            result_a = None
            exception_a = e
        # Special case: compiled versions don't match the error type exactly.
        if ((exception_e is None) != (exception_a is None)) or not ignore_exc_types:
            if exception_a is not None and exception_e is None:
                raise exception_a
            self.assertIs(type(exception_e), type(exception_a), f"\n{exception_e=}\n{exception_a=}")

        if exception_e is None:
            flattened_e, spec_e = tree_flatten(result_e)
            flattened_a, spec_a = tree_flatten(result_a)

            self.assertEqual(
                spec_e, spec_a, "Both functions must return a result with the same tree structure."
            )
            for value_e, value_a in zip(flattened_e, flattened_a, strict=True):
                value_e = _as_interleaved(value_e)
                value_a = _as_interleaved(value_a)

                self.assertEqual(value_e, value_a, *args, **kwargs)
