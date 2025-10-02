from __future__ import annotations

from pathlib import Path

import onnx
import torch
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
    model.opset_import[0].version = 17
    onnx.save(model, path)


def test_build_example_args_with_warmup_overrides(tmp_path):
    model_path = tmp_path / "static.onnx"
    _write_identity_model(model_path, [1, 3, 16])
    model = onnx.load(str(model_path))

    example_cfg = run.ExampleInputConfig(
        default_fill="zeros",
        overrides={
            "input": {
                "shape": [1, 3, 32],
                "warmup_shape": [1, 3, 4],
                "dim_labels": ["batch", "channels", "sequence_length"],
            }
        },
        scales={"sequence_length": 24},
    )

    config = run.RunnerConfig(inputs=[], example_inputs=example_cfg)
    runtime_args, warmup_args = run._build_example_and_warmup_args(model, config)

    assert runtime_args[0].shape == torch.Size([1, 3, 24])
    assert warmup_args[0].shape == torch.Size([1, 3, 4])


def test_element_cap_clamps_large_tensors(tmp_path):
    model_path = tmp_path / "big.onnx"
    _write_identity_model(model_path, [1, 3, 512])
    model = onnx.load(str(model_path))

    example_cfg = run.ExampleInputConfig(
        default_fill="zeros",
        overrides={"input": {"shape": [1, 3, 512]}},
        max_total_elements=1_000,
    )

    config = run.RunnerConfig(inputs=[], example_inputs=example_cfg)
    runtime_args, warmup_args = run._build_example_and_warmup_args(model, config)

    assert torch.prod(torch.tensor(runtime_args[0].shape)).item() <= 1_000
    assert (
        torch.prod(torch.tensor(warmup_args[0].shape)).item()
        <= example_cfg.warmup_max_total_elements
    )


def test_parse_scale_overrides_accepts_multiple_entries():
    overrides = run._parse_scale_overrides(["seq=512", "batch=8"])
    assert overrides == {"seq": 512, "batch": 8}
