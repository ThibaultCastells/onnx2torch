import numpy as np
import torch

from onnx2torch.node_converters.slice import OnnxSlice
from onnx2torch.utils.shape_warmup import shape_warmup


def test_shape_warmup_handles_index_select_out_of_bounds() -> None:
    """Index select should not crash during shape warmup when indices exceed bounds."""

    input_tensor = torch.randn(1, 3)
    indices = torch.tensor([0, 5], dtype=torch.int64)

    with shape_warmup():
        result = torch.index_select(input_tensor, 0, indices)

    assert result.shape == (2, 3)
    assert torch.count_nonzero(result) == 0


def test_shape_warmup_handles_view_shape_mismatch() -> None:
    """View should return zero placeholder even when element counts differ."""

    input_tensor = torch.randn(2, 2)
    target_shape = (2, 2, 2)

    with shape_warmup():
        result = torch.ops.aten.view.default(input_tensor, target_shape)

    assert result.shape == target_shape
    assert result.dtype == input_tensor.dtype
    assert torch.count_nonzero(result) == 0


def test_shape_warmup_handles_expand_broadcast() -> None:
    """Expand should broadcast to the requested shape without real data movement."""

    input_tensor = torch.randn(1, 1, 4)
    target_shape = (2, 2, 4)

    with shape_warmup():
        result = torch.ops.aten.expand.default(input_tensor, target_shape)

    assert result.shape == target_shape
    assert result.dtype == input_tensor.dtype
    assert torch.count_nonzero(result) == 0


def test_shape_warmup_expand_replaces_non_positive_dims() -> None:
    """Zero and -1 dimensions should fall back to meaningful sizes."""

    input_tensor = torch.randn(2, 4)
    target_shape = (0, -1)

    with shape_warmup():
        result = torch.ops.aten.expand.default(input_tensor, target_shape)

    assert result.shape == (2, 4)


def test_shape_warmup_mul_handles_unbroadcastable_shapes() -> None:
    """Binary operations should succeed even when placeholder shapes mismatch."""

    lhs = torch.randn(2, 3)
    rhs = torch.randn(4, 1)

    with shape_warmup():
        result = torch.mul(lhs, rhs)

    assert result.shape == (4, 3)
    assert torch.count_nonzero(result) == 0


def test_shape_warmup_slice_accepts_zero_step() -> None:
    """Slice should treat zero steps as one during warmup to avoid failures."""

    module = OnnxSlice(
        constant_axes=np.array([1], dtype=np.int64),
        constant_steps=np.array([0], dtype=np.int64),
    )

    input_tensor = torch.randn(1, 4)
    starts = torch.tensor([0], dtype=torch.int64)
    ends = torch.tensor([2], dtype=torch.int64)
    axes = torch.tensor([1], dtype=torch.int64)
    steps = torch.tensor([0], dtype=torch.int64)

    with shape_warmup():
        result = module(input_tensor, starts, ends, axes, steps)

    assert result.dim() == input_tensor.dim()
