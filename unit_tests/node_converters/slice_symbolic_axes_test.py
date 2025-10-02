import numpy as np
import pytest
import torch

from onnx2torch.node_converters.slice import OnnxSlice

try:  # pragma: no cover - ShapeEnv may be unavailable on older torch builds
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
except ImportError:  # pragma: no cover - keep skipif working when ShapeEnv is absent
    ShapeEnv = None  # type: ignore[assignment]


sympy = pytest.importorskip("sympy")


@pytest.mark.skipif(ShapeEnv is None, reason="ShapeEnv is unavailable")
def test_slice_symbolic_axes_fallback_to_constants() -> None:  # pylint: disable=missing-function-docstring
    module = OnnxSlice(constant_axes=np.array([0], dtype=np.int64))

    assert ShapeEnv is not None  # for mypy; guarded by skipif
    shape_env = ShapeEnv()
    sym_axis = shape_env.create_symintnode(sympy.Symbol("axis"), hint=0)

    input_tensor = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    starts = torch.tensor([0], dtype=torch.int64)
    ends = torch.tensor([2], dtype=torch.int64)
    steps = torch.tensor([1], dtype=torch.int64)

    output = module(input_tensor, starts, ends, (sym_axis,), steps)
    expected = input_tensor[0:2]

    assert torch.allclose(output, expected)
