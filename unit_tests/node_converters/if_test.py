import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto
from onnx import helper
from onnx import numpy_helper

from unittest.mock import patch

from onnx2torch.converter import convert
from onnx2torch.node_converters.control_flow import OnnxIf
from onnx2torch.node_converters.control_flow import UncapturedHigherOrderOpError
from onnx2torch.node_converters.slice import OnnxSlice
from ..utils.common import calc_ort_outputs
from ..utils.common import calc_torch_outputs

try:  # pragma: no cover - torch.export may be absent on older builds
    from torch.export import export as export_program
except ImportError:  # pragma: no cover - keep skip working without torch.export
    export_program = None  # type: ignore[assignment]


def _make_if_model() -> onnx.ModelProto:
    cond_info = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])
    out_info = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2])
    pre_value_info = helper.make_tensor_value_info("pre_in", TensorProto.FLOAT, [2])

    then_graph = helper.make_graph(
        nodes=[helper.make_node("Mul", ["pre_in", "two"], ["then_out"])],
        name="then_branch",
        inputs=[],
        outputs=[
            helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [2]),
        ],
        value_info=[
            helper.make_tensor_value_info("pre_in", TensorProto.FLOAT, [2]),
        ],
    )

    else_graph = helper.make_graph(
        nodes=[helper.make_node("Sub", ["pre_in", "y"], ["else_out"])],
        name="else_branch",
        inputs=[],
        outputs=[
            helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [2]),
        ],
        value_info=[
            helper.make_tensor_value_info("pre_in", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [2]),
        ],
    )

    if_node = helper.make_node(
        "If",
        inputs=["cond"],
        outputs=["out"],
        then_branch=then_graph,
        else_branch=else_graph,
    )

    add_node = helper.make_node("Add", ["x", "y"], ["pre_in"])

    graph = helper.make_graph(
        name="if_test_graph",
        nodes=[add_node, if_node],
        inputs=[cond_info, x_info, y_info],
        outputs=[out_info],
        value_info=[pre_value_info],
        initializer=[
            numpy_helper.from_array(
                np.array([2.0, 2.0], dtype=np.float32),
                name="two",
            ),
        ],
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 16)],
        producer_name="onnx2torch_if_test",
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_if_slice_model() -> onnx.ModelProto:
    cond_info = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    data_info = helper.make_tensor_value_info("data", TensorProto.FLOAT, [4])
    out_info = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2])

    then_graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Slice",
                ["data", "slice_starts", "slice_ends", "slice_axes"],
                ["then_out"],
            ),
        ],
        name="then_branch_slice",
        inputs=[],
        outputs=[
            helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [2]),
        ],
        value_info=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, [4]),
            helper.make_tensor_value_info("slice_starts", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("slice_ends", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("slice_axes", TensorProto.INT64, [1]),
        ],
    )

    else_graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Slice",
                ["data", "slice_starts", "slice_ends", "slice_axes"],
                ["else_out"],
            ),
        ],
        name="else_branch_slice",
        inputs=[],
        outputs=[
            helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [2]),
        ],
        value_info=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, [4]),
            helper.make_tensor_value_info("slice_starts", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("slice_ends", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("slice_axes", TensorProto.INT64, [1]),
        ],
    )

    if_node = helper.make_node(
        "If",
        inputs=["cond"],
        outputs=["out"],
        name="SliceIf",
        then_branch=then_graph,
        else_branch=else_graph,
    )

    graph = helper.make_graph(
        name="if_slice_graph",
        nodes=[if_node],
        inputs=[cond_info, data_info],
        outputs=[out_info],
        initializer=[
            numpy_helper.from_array(
                np.array([0], dtype=np.int64),
                name="slice_starts",
            ),
            numpy_helper.from_array(
                np.array([2], dtype=np.int64),
                name="slice_ends",
            ),
            numpy_helper.from_array(
                np.array([0], dtype=np.int64),
                name="slice_axes",
            ),
        ],
        value_info=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, [4]),
            helper.make_tensor_value_info("slice_starts", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("slice_ends", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("slice_axes", TensorProto.INT64, [1]),
        ],
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
        producer_name="onnx2torch_if_slice_test",
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


@pytest.mark.parametrize("cond_value", [True, False])
def test_if_converter_matches_onnx_runtime(cond_value: bool) -> None:
    model = _make_if_model()
    inputs = {
        "cond": np.array(cond_value, dtype=np.bool_),
        "x": np.array([1.0, -2.0], dtype=np.float32),
        "y": np.array([0.5, 3.0], dtype=np.float32),
    }
    ort_outputs = calc_ort_outputs(model, inputs)
    torch_outputs = calc_torch_outputs(model, inputs)
    torch_outputs = (
        list(torch_outputs)
        if isinstance(torch_outputs, (list, tuple))
        else [torch_outputs]
    )
    for expected, actual in zip(ort_outputs, torch_outputs, strict=True):
        np.testing.assert_allclose(expected, actual, atol=1e-5)


def test_if_branch_hoists_parent_initializers() -> None:
    model = _make_if_slice_model()
    torch_module = convert(model).eval()

    inputs = {
        "cond": np.array(True, dtype=np.bool_),
        "data": np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32),
    }
    ort_true = calc_ort_outputs(model, inputs)
    torch_true = calc_torch_outputs(model, inputs)
    torch_true = (
        list(torch_true) if isinstance(torch_true, (list, tuple)) else [torch_true]
    )
    for expected, actual in zip(ort_true, torch_true, strict=True):
        np.testing.assert_allclose(expected, actual, atol=1e-5)

    inputs["cond"] = np.array(False, dtype=np.bool_)
    ort_false = calc_ort_outputs(model, inputs)
    torch_false = calc_torch_outputs(model, inputs)
    torch_false = (
        list(torch_false) if isinstance(torch_false, (list, tuple)) else [torch_false]
    )
    for expected, actual in zip(ort_false, torch_false, strict=True):
        np.testing.assert_allclose(expected, actual, atol=1e-5)

    onnx_if_modules = [
        module for module in torch_module.modules() if isinstance(module, OnnxIf)
    ]
    assert onnx_if_modules, "Converted module should include OnnxIf"
    onnx_if = onnx_if_modules[0]

    for name in ("slice_starts", "slice_ends", "slice_axes"):
        assert name not in onnx_if._captured_inputs

    for branch_module in (onnx_if.then_branch_module, onnx_if.else_branch_module):
        slice_modules = [
            module for module in branch_module.modules() if isinstance(module, OnnxSlice)
        ]
        assert slice_modules, "Branch module must include OnnxSlice"
        for slice_module in slice_modules:
            assert slice_module._constant_axes == (0,)
            assert slice_module._constant_starts == (0,)
            assert slice_module._constant_ends == (2,)


def test_if_fallback_handles_uncaptured_higher_order_op_error() -> None:
    model = _make_if_model()
    torch_module = convert(model).eval()

    inputs = {
        "cond": np.array(True, dtype=np.bool_),
        "x": np.array([1.0, -2.0], dtype=np.float32),
        "y": np.array([0.5, 3.0], dtype=np.float32),
    }
    ort_outputs = calc_ort_outputs(model, inputs)
    torch_inputs = (
        torch.tensor(inputs["cond"]),
        torch.from_numpy(inputs["x"]),
        torch.from_numpy(inputs["y"]),
    )

    with patch("torch.cond", side_effect=UncapturedHigherOrderOpError("graph break")):
        module_outputs = torch_module(*torch_inputs)

    module_outputs = (
        list(module_outputs)
        if isinstance(module_outputs, (list, tuple))
        else [module_outputs]
    )

    for expected, actual in zip(ort_outputs, module_outputs, strict=True):
        np.testing.assert_allclose(
            expected,
            actual.detach().cpu().numpy(),
            atol=1e-5,
        )


@pytest.mark.skipif(export_program is None, reason="torch.export is unavailable")
def test_if_converter_supports_torch_export() -> None:
    model = _make_if_model()
    torch_module = convert(model).eval()
    sample_cond = torch.tensor(True)
    sample_x = torch.randn(2)
    sample_y = torch.randn(2)

    exported = export_program(torch_module, (sample_cond, sample_x, sample_y))
    assert exported is not None
