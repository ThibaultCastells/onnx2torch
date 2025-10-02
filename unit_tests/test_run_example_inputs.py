from __future__ import annotations

from pathlib import Path

import onnx
import torch
from onnx import TensorProto, helper

import run


def _write_identity_model(path: Path, input_shape, dtype=TensorProto.FLOAT):
    input_tensor = helper.make_tensor_value_info("input", dtype, input_shape)
    output_tensor = helper.make_tensor_value_info("output", dtype, input_shape)
    node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
    graph = helper.make_graph([node], "identity", [input_tensor], [output_tensor])
    model = helper.make_model(graph, producer_name="onnx2torch-test")
    model.opset_import[0].version = 17
    onnx.save(model, path)


def test_build_example_args_use_onnx_declared_shapes(tmp_path):
    model_path = tmp_path / "static.onnx"
    _write_identity_model(model_path, [1, 3, 16])
    model = onnx.load(str(model_path))

    example_cfg = run.ExampleInputConfig(
        default_fill="ones",
    )

    config = run.RunnerConfig(inputs=[], example_inputs=example_cfg)
    runtime_args, warmup_args = run._build_example_and_warmup_args(model, config)

    assert runtime_args[0].shape == torch.Size([1, 3, 16])
    assert warmup_args[0].shape == torch.Size([1, 3, 16])
    assert torch.all(runtime_args[0] == 1)


def test_build_example_args_without_warmup(tmp_path):
    model_path = tmp_path / "static.onnx"
    _write_identity_model(model_path, [2, 4])
    model = onnx.load(str(model_path))

    example_cfg = run.ExampleInputConfig(
        enable_warmup=False,
    )

    config = run.RunnerConfig(inputs=[], example_inputs=example_cfg)
    runtime_args, warmup_args = run._build_example_and_warmup_args(model, config)

    assert runtime_args[0].shape == torch.Size([2, 4])
    assert warmup_args == tuple()


def test_build_example_args_support_bool_inputs(tmp_path):
    model_path = tmp_path / "bool.onnx"
    _write_identity_model(model_path, [1, 5], dtype=TensorProto.BOOL)
    model = onnx.load(str(model_path))

    config = run.RunnerConfig(inputs=[], example_inputs=run.ExampleInputConfig())
    runtime_args, warmup_args = run._build_example_and_warmup_args(model, config)

    assert runtime_args[0].dtype is torch.bool
    assert warmup_args[0].dtype is torch.bool
