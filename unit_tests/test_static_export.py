from __future__ import annotations

from pathlib import Path

import onnx
import torch
from onnx import TensorProto, helper

import run


def _write_identity_model(path: Path) -> None:
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
    graph = helper.make_graph([node], "identity", [input_tensor], [output_tensor])
    model = helper.make_model(graph, producer_name="onnx2torch-test")
    model.opset_import[0].version = 17
    onnx.save(model, path)


def test_exported_program_has_no_sym_constrain(tmp_path):
    onnx_path = tmp_path / "identity.onnx"
    _write_identity_model(onnx_path)

    task = run.ModelTask(source=onnx_path, destination=tmp_path / "identity.pt2")
    config = run.RunnerConfig(inputs=[], example_inputs=run.ExampleInputConfig())
    run_context = run.RunContext(directory=tmp_path, verbosity=0)

    run._export_model(task, config, run_context)

    exported = torch.export.load(str(task.destination))
    assert not any(
        node.op == "call_function"
        and node.target is torch.ops.aten.sym_constrain_range_for_size.default
        for node in exported.graph_module.graph.nodes
    )
