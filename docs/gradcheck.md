# Gradient checking

## Mechanism

The gradient checking is done implicitly, by supporting all the required ops and letting PyTorch
take care of the rest.

## Tests

The gradient checking tests are currently placed in [`src/complex_tensor/test/test_ops.py::TestComplexBwdGradients`](https://github.com/openteams-ai/pytorch-complex-tensor/blob/main/src/complex_tensor/test/test_ops.py).

The gradient check tests check the consistency between numerical differentiation versus analytical formulas as implemented by the `complex_tensor.ComplexTensor` class. They do not test numerical accuracy of the gradients vs the gradients calculated in eager mode. The tests use a `TorchDispatchMode` to map all `torch.Tensor`s with a complex dtype to
`complex_tensor.ComplexTensor`s. The op is then run as usual, going through the dispatching machinery,
hitting `complex_tensor.ComplexTensor.__torch_dispatch__` as well and then dispatching to the correct op.

The results are returned as a `torch.Tensor` for consistency.

## Outstanding Issues

### Edge-case Behavior

Quite a lot of the gradient checking tests are skipped due to edge-case behavior and slight numerical
inconsistencies. By edge-case behavior, we mean the behavior when an input contaning `NaN`s or
infinities (positive or negative) are passed in.

An effort was made to mirror implementations found in LLVM's `libc++` for numerical accuracy. However,
matching edge-case behavior was difficult for a number of reasons:

1. The implementations contained a lot of `if-elseif-else` statements. In the PyTorch world, that'd
   transfer into `torch.where` calls, which would lead to a lot of temporaries (in eager mode) or branches
   (in compiled mode), which would hinder the performance on GPUs.
2. There were a lot of these cases per-op.

However, we did find it was a toss up as to which was more consistent with IEEE-754 between
`torch.Tensor` and `complex_tensor.ComplexTensor`.

### Spurious Failures

Right now, the samples used for gradient checking are drawn randomly from a given distribution
per-op inside `torch.testing._internal.common_utils.TestGradients._grad_test_helper`, which we
re-use in our tests. In some cases, edge-cases are never hit while in others, they are, leading
to spurious failures.

We did find that allowing [`equal_nan=True` inside gradcheck](https://github.com/pytorch/pytorch/pull/164928)
helped with some failures, but not with others. `torch.Tensor` with complex dtypes passes all these
tests with flying colors even with `equal_nan=False`.

### Re-definition of Backward Functions

For very few ops, namely `tanh` and `diagonal` so far, the backward function had to be re-defined, despite
decompositions existing inside PyTorch. We're unsure why this is the case, perhaps these aren't properly
registered at some level, or overridden.
