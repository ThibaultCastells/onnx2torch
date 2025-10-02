"""Tests for FakeTensor-specific behaviour of the reshape converter."""

from __future__ import annotations

from unittest import mock

import pytest
import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from onnx2torch.node_converters.reshape import OnnxReshape

try:  # pragma: no cover - FakeTensor may be unavailable on older torch builds
    from torch._subclasses.fake_tensor import FakeTensor
    from torch._subclasses.fake_tensor import FakeTensorMode
except ImportError:  # pragma: no cover - ensure tests skip cleanly
    FakeTensor = ()  # type: ignore[assignment]
    FakeTensorMode = None


@pytest.mark.skipif(
    FakeTensorMode is None, reason="FakeTensor is not available on this PyTorch build"
)
def test_cached_shape_used_during_fake_tensor_execution() -> None:
    """Warmup should cache shapes that FakeTensor execution reuses."""

    module = OnnxReshape()
    input_real = torch.zeros((1, 16, 16, 128))
    shape_tensor = torch.tensor([1, 16, 16, 128], dtype=torch.int64)

    # Warmup with concrete tensors to populate the cache.
    module(input_real, shape_tensor)
    assert module._cached_target_shape == (1, 16, 16, 128)

    shape_env = ShapeEnv()
    mode = FakeTensorMode(shape_env=shape_env, static_shapes=True)
    with mode:
        fake_input = torch.zeros((1, 16, 16, 128))
        fake_shape = torch.tensor([1, 16, 16, 128], dtype=torch.int64)

        assert isinstance(fake_input, FakeTensor)

        with mock.patch(
            "onnx2torch.node_converters.reshape.shape_tensor_to_sequence",
            side_effect=AssertionError("shape_tensor_to_sequence should not run"),
        ):
            result = module(fake_input, fake_shape)

        assert isinstance(result, FakeTensor)
        assert result.shape == fake_input.shape
