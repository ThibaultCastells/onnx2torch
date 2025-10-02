import pytest
import torch

from onnx2torch.node_converters.expand import OnnxExpand

try:  # pragma: no cover - FakeTensor may be unavailable on some torch builds
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
except ImportError:  # pragma: no cover - keep skipif working when FakeTensor is absent
    FakeTensorMode = None  # type: ignore[assignment]
    ShapeEnv = None  # type: ignore[assignment]

try:  # pragma: no cover - torch.export may not exist on some torch releases
    from torch.export import Dim
    from torch.export import export as export_program
except ImportError:  # pragma: no cover - skip export-specific tests when unavailable
    Dim = None  # type: ignore[assignment]
    export_program = None  # type: ignore[assignment]


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


@pytest.mark.skipif(export_program is None, reason="torch.export is unavailable")
def test_expand_traces_with_symbolic_shape() -> None:  # pylint: disable=missing-function-docstring
    class ExportModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.expand = OnnxExpand()

        def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            dims = [
                torch.ops.aten.scalar_tensor.default(
                    torch.ops.aten.sym_size.int(input_tensor, 0),
                    dtype=torch.int64,
                    device=input_tensor.device,
                ),
                torch.tensor(1, dtype=torch.int64, device=input_tensor.device),
                torch.ops.aten.scalar_tensor.default(
                    torch.ops.aten.sym_size.int(input_tensor, 1),
                    dtype=torch.int64,
                    device=input_tensor.device,
                ),
            ]
            shape = torch.ops.aten.stack.default(dims, 0)
            return self.expand(input_tensor, shape)

    model = ExportModel().eval()
    sample = torch.randn(2, 3)

    assert Dim is not None  # mypy hint; guarded by skip
    dynamic_shapes = ({0: Dim("batch"), 1: Dim("feature")},)

    exported = export_program(model, (sample,), dynamic_shapes=dynamic_shapes)
    assert exported is not None
