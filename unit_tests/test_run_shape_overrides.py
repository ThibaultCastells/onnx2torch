from __future__ import annotations

from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

import run


def _write_identity_model(path: Path, input_shape):
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, input_shape
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, input_shape
    )
    node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
    graph = helper.make_graph([node], "identity", [input_tensor], [output_tensor])
    model = helper.make_model(graph, producer_name="onnx2torch-test")
    onnx.save(model, path)


def test_prepare_model_requires_shape_overrides(tmp_path):
    model_path = tmp_path / "dynamic_input.onnx"
    _write_identity_model(model_path, ["batch", 3])

    task = run.ModelTask(source=model_path, destination=tmp_path / "out.pt2")
    config = run.RunnerConfig(inputs=[])

    with pytest.raises(run.ShapePreparationError) as excinfo:
        run._prepare_model_with_shapes(task, config)

    assert "input" in str(excinfo.value)


def test_prepare_model_applies_shape_overrides(tmp_path):
    model_path = tmp_path / "dynamic_input.onnx"
    _write_identity_model(model_path, ["batch", 3])

    task = run.ModelTask(source=model_path, destination=tmp_path / "out.pt2")
    config = run.RunnerConfig(
        inputs=[],
        input_shapes={model_path.resolve(): {"input": (2, 3)}},
    )

    prepared = run._prepare_model_with_shapes(task, config)
    assert run._input_names_missing_static_shapes(prepared) == set()


def test_prepare_model_detects_missing_override_for_specific_inputs(tmp_path):
    model_path = tmp_path / "two_inputs.onnx"
    input_shapes = {
        "input_a": ["batch", 3],
        "input_b": ["batch", 5],
    }
    inputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
        for name, shape in input_shapes.items()
    ]
    outputs = [
        helper.make_tensor_value_info(name + "_out", TensorProto.FLOAT, shape)
        for name, shape in input_shapes.items()
    ]
    nodes = [
        helper.make_node(
            "Identity",
            inputs=[name],
            outputs=[name + "_out"],
            name=f"identity_{name}",
        )
        for name in input_shapes.keys()
    ]
    graph = helper.make_graph(nodes, "two_identity", inputs, outputs)
    model = helper.make_model(graph, producer_name="onnx2torch-test")
    onnx.save(model, model_path)

    task = run.ModelTask(source=model_path, destination=tmp_path / "out.pt2")
    config = run.RunnerConfig(
        inputs=[],
        input_shapes={model_path.resolve(): {"input_a": (4, 3)}},
    )

    with pytest.raises(run.ShapePreparationError) as excinfo:
        run._prepare_model_with_shapes(task, config)

    assert "input_b" in str(excinfo.value)


def test_prepare_model_falls_back_when_onnxsim_ir_version_error(tmp_path, monkeypatch):
    model_path = tmp_path / "dynamic_input.onnx"
    _write_identity_model(model_path, ["batch", 3])

    task = run.ModelTask(source=model_path, destination=tmp_path / "out.pt2")
    config = run.RunnerConfig(
        inputs=[],
        input_shapes={model_path.resolve(): {"input": (2, 3)}},
    )

    def _raise_ir_version_error(*_args, **_kwargs):
        raise RuntimeError("The model does not have an ir_version set properly.")

    monkeypatch.setattr(run, "_onnxsim_simplify", _raise_ir_version_error)

    prepared = run._prepare_model_with_shapes(task, config)
    assert run._input_names_missing_static_shapes(prepared) == set()
