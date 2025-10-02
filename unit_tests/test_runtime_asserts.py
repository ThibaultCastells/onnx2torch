import io
import zipfile

import pytest
import torch

from onnx2torch.node_converters.expand import OnnxExpand

from run import _strip_scalar_runtime_asserts


@pytest.mark.skipif(
    getattr(torch.export, "export", None) is None, reason="torch.export is unavailable"
)
def test_strip_scalar_runtime_asserts_removes_sym_bool() -> None:
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

    _strip_scalar_runtime_asserts(exported)

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
