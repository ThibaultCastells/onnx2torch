import pytest
import torch

from onnx2torch.utils.shape_utils import sequence_to_symint_tuple
from onnx2torch.utils.shape_warmup import shape_warmup

try:  # pragma: no cover - FakeTensor may be unavailable on older torch builds
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
except ImportError:  # pragma: no cover - keep skipif working without FakeTensor
    FakeTensorMode = None  # type: ignore[assignment]
    ShapeEnv = None  # type: ignore[assignment]


@pytest.mark.skipif(FakeTensorMode is None, reason="FakeTensorMode is unavailable")
def test_sequence_to_symint_tuple_preserves_fake_tensor_symbolics() -> None:
    """Ensure FakeTensor scalars remain symbolic and retain their runtime hints."""

    assert ShapeEnv is not None  # for mypy; guarded by skipif
    mode = FakeTensorMode(shape_env=ShapeEnv())
    with mode:
        tensor = torch.tensor([0, 1], dtype=torch.int64)

    result = sequence_to_symint_tuple(tensor)

    assert len(result) == 2
    for value in result:
        assert hasattr(value, "node")
        assert not isinstance(value, int)


def test_sequence_to_symint_tuple_respects_shape_warmup() -> None:
    """Constants should retain their scalar values during shape warmup."""

    tensor = torch.tensor([0, 64, 3, 1], dtype=torch.int64)

    with shape_warmup():
        result = sequence_to_symint_tuple(tensor)

    assert result == (0, 64, 3, 1)
