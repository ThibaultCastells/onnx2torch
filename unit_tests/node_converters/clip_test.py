from typing import Optional
from typing import Tuple

import numpy as np
import onnx

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_clip(
    input_shape: Tuple[int, int, int, int],
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    **kwargs,
) -> None:
    x_range = 2 * max_value if max_value is not None else 5
    x = np.random.uniform(low=-x_range, high=x_range, size=input_shape).astype(
        np.float32
    )
    test_inputs = {"x": x}

    initializers = {}
    if min_value is not None:
        initializers["min"] = np.array(min_value, dtype=np.float32)

    if max_value is not None:
        initializers["max"] = np.array(max_value, dtype=np.float32)

    node = onnx.helper.make_node(
        op_type="Clip",
        inputs=list(test_inputs) + list(initializers),
        outputs=["y"],
        **kwargs,
    )
    model = make_model_from_nodes(
        nodes=node, initializers=initializers, inputs_example=test_inputs
    )
    check_onnx_model(model, test_inputs)


def _test_clip_opset9(
    input_shape: Tuple[int, int, int, int],
    **kwargs,
) -> None:
    x = np.random.uniform(low=-10.0, high=10.0, size=input_shape).astype(np.float32)
    test_inputs = {"x": x}

    node = onnx.helper.make_node(
        op_type="Clip",
        inputs=list(test_inputs),
        outputs=["y"],
        **kwargs,
    )
    model = make_model_from_nodes(
        nodes=node, initializers={}, inputs_example=test_inputs, opset_version=9
    )
    check_onnx_model(model, test_inputs)


def test_clip() -> None:  # pylint: disable=missing-function-docstring
    _test_clip(input_shape=(2, 3, 16, 16), min_value=0.0, max_value=6.0)
    _test_clip(input_shape=(2, 3, 16, 16), min_value=0.0)
    _test_clip(input_shape=(2, 3, 16, 16), min_value=-1.5, max_value=2.5)
    _test_clip_opset9(input_shape=(2, 3, 16, 16), min=0.0, max=6.0)
    _test_clip_opset9(input_shape=(2, 3, 16, 16), min=0.0)
    _test_clip_opset9(input_shape=(2, 3, 16, 16), min=-1.7, max=2.8)


def test_clip_with_tensor_initializers() -> None:  # pylint: disable=missing-function-docstring
    input_shape = (2, 3, 8, 8)
    x = np.random.uniform(low=-4.0, high=4.0, size=input_shape).astype(np.float32)
    min_tensor = np.random.uniform(low=-1.0, high=0.0, size=(1, 3, 1, 1)).astype(
        np.float32
    )
    max_tensor = min_tensor + np.random.uniform(
        low=0.5, high=2.0, size=(1, 3, 1, 1)
    ).astype(np.float32)

    node = onnx.helper.make_node(
        op_type="Clip",
        inputs=["x", "min", "max"],
        outputs=["y"],
    )
    model = make_model_from_nodes(
        nodes=node,
        initializers={"min": min_tensor, "max": max_tensor},
        inputs_example={"x": x},
    )
    check_onnx_model(model, {"x": x})


def test_clip_with_dynamic_min() -> None:  # pylint: disable=missing-function-docstring
    input_shape = (1, 4, 16, 16)
    x = np.random.uniform(low=-3.0, high=3.0, size=input_shape).astype(np.float32)
    min_tensor = np.random.uniform(low=-2.0, high=1.0, size=(1, 4, 1, 1)).astype(
        np.float32
    )

    test_inputs = {"x": x, "min": min_tensor}
    node = onnx.helper.make_node(
        op_type="Clip",
        inputs=["x", "min"],
        outputs=["y"],
    )
    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
    )
    check_onnx_model(model, test_inputs)


def test_clip_with_dynamic_max() -> None:  # pylint: disable=missing-function-docstring
    input_shape = (1, 4, 16, 16)
    x = np.random.uniform(low=-3.0, high=3.0, size=input_shape).astype(np.float32)
    max_tensor = np.random.uniform(low=0.5, high=3.0, size=(1, 4, 1, 1)).astype(
        np.float32
    )

    test_inputs = {"x": x, "max": max_tensor}
    node = onnx.helper.make_node(
        op_type="Clip",
        inputs=["x", "", "max"],
        outputs=["y"],
    )
    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
    )
    check_onnx_model(model, test_inputs)


def test_clip_with_dynamic_min_and_max() -> None:  # pylint: disable=missing-function-docstring
    input_shape = (1, 2, 32, 32)
    x = np.random.uniform(low=-5.0, high=5.0, size=input_shape).astype(np.float32)
    min_tensor = np.random.uniform(low=-3.0, high=0.0, size=input_shape).astype(
        np.float32
    )
    span = np.random.uniform(low=0.1, high=3.0, size=input_shape).astype(np.float32)
    max_tensor = min_tensor + span

    test_inputs = {"x": x, "min": min_tensor, "max": max_tensor}
    node = onnx.helper.make_node(
        op_type="Clip",
        inputs=["x", "min", "max"],
        outputs=["y"],
    )
    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
    )
    check_onnx_model(model, test_inputs)
