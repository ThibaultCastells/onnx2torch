import pytest
import torch

from onnx2torch.node_converters.expand import OnnxExpand

try:  # pragma: no cover - FakeTensor may be unavailable on some torch builds
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
except ImportError:  # pragma: no cover - keep skipif working when FakeTensor is absent
    FakeTensorMode = None  # type: ignore[assignment]
    ShapeEnv = None  # type: ignore[assignment]


@pytest.mark.skipif(FakeTensorMode is None, reason="FakeTensorMode is unavailable")
def test_expand_accepts_fake_tensor_shape() -> None:  # pylint: disable=missing-function-docstring
    module = OnnxExpand()

    assert ShapeEnv is not None  # for mypy; guarded by skipif
    mode = FakeTensorMode(shape_env=ShapeEnv())
    with mode:
        input_tensor = torch.randn(1, 4)
        shape = torch.tensor([2, 1, 4], dtype=torch.int64)
        _ = module(input_tensor, shape)

    real_output = module(torch.randn(1, 4), torch.tensor([2, 1, 4], dtype=torch.int64))
    assert real_output.shape == (2, 1, 4)
