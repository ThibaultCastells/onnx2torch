import io
import zipfile

import pytest
import torch

from onnx2torch.node_converters.expand import OnnxExpand

from run import _strip_runtime_guards


@pytest.mark.skipif(
    getattr(torch.export, "export", None) is None, reason="torch.export is unavailable"
)
def test_strip_runtime_guards_removes_scalar_asserts() -> None:
    class Model(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            shape = torch.tensor([2, 1, 4], dtype=torch.int64)
            return OnnxExpand()(x, shape)

    args = (torch.randn(2, 3),)
    exported = torch.export.export(
        Model().eval(),
        args,
        strict=True,
    )

    assert any(
        node.target is torch.ops.aten._assert_scalar.default
        for node in exported.graph_module.graph.nodes
    )

    _strip_runtime_guards(exported)

    assert all(
        node.target is not torch.ops.aten._assert_scalar.default
        for node in exported.graph_module.graph.nodes
    )

    from torch.export import save as export_save

    buffer = io.BytesIO()
    export_save(exported, buffer)
    archive = zipfile.ZipFile(io.BytesIO(buffer.getvalue()))
    model_json = archive.read("archive/models/model.json").decode("utf-8")
    assert '"sym_bool_values": {}' in model_json


def test_strip_runtime_guards_removes_sym_constrain() -> None:
    graph = torch.fx.Graph()
    tensor_in = graph.placeholder("x")
    item = graph.call_function(torch.ops.aten.item.default, (tensor_in,))
    graph.call_function(
        torch.ops.aten.sym_constrain_range_for_size.default,
        (item,),
        {"min": 1, "max": 64},
    )
    graph.output((tensor_in,))

    module = torch.fx.GraphModule(torch.nn.Module(), graph)

    class DummyExport:
        def __init__(self, graph_module: torch.fx.GraphModule) -> None:
            self.graph_module = graph_module

    exported = DummyExport(module)

    assert any(
        node.target is torch.ops.aten.sym_constrain_range_for_size.default
        for node in exported.graph_module.graph.nodes
    )

    _strip_runtime_guards(exported)

    assert all(
        node.target is not torch.ops.aten.sym_constrain_range_for_size.default
        for node in exported.graph_module.graph.nodes
    )
    assert all(
        node.target is not torch.ops.aten.item.default
        for node in exported.graph_module.graph.nodes
    )
