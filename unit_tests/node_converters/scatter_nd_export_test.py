import pytest
import torch

from onnx2torch.node_converters.scatter_nd import OnnxScatterND
from onnx2torch.node_converters.scatter_nd import ReductionOnnxAttr

try:  # pragma: no cover - torch.export may not be available on older torch versions
    from torch.export import Dim
    from torch.export import export as export_program
except ImportError:  # pragma: no cover - keep skipif working when export is absent
    Dim = None  # type: ignore[assignment]
    export_program = None  # type: ignore[assignment]


@pytest.mark.skipif(export_program is None, reason="torch.export is unavailable")
def test_scatter_nd_traces_with_symbolic_shape() -> None:  # pylint: disable=missing-function-docstring
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.op = OnnxScatterND(reduction=ReductionOnnxAttr.NONE)

        def forward(  # type: ignore[override]
            self,
            data: torch.Tensor,
            indices: torch.Tensor,
            updates: torch.Tensor,
        ) -> torch.Tensor:
            return self.op(data, indices, updates)

    assert Dim is not None  # mypy assertion; guarded by skipif
    assert export_program is not None  # mypy assertion; guarded by skipif

    model = Model().eval()

    data = torch.randn(2, 3, 4)
    indices = torch.tensor([[[0, 1]], [[1, 2]]], dtype=torch.int64)
    updates = torch.randn(2, 1, 4)

    dynamic_shapes = (
        {0: Dim("batch"), 1: Dim("row"), 2: Dim("feature")},
        {0: Dim("update_count")},
        {0: Dim("update_count"), 2: Dim("feature")},
    )

    exported = export_program(
        model, (data, indices, updates), dynamic_shapes=dynamic_shapes
    )
    assert exported is not None

    graph_module = exported.graph_module
    inplace_ops = {
        str(node.target) for node in graph_module.graph.nodes if hasattr(node, "target")
    }
    assert "aten.mul_.Tensor" not in inplace_ops
    assert "aten.index_put_.default" not in inplace_ops
