from typing import Any, Dict, List, Optional

import numpy as np
import onnx
import pytest
from onnx.helper import make_tensor_value_info
try:  # ONNX >=1.17 exposes mapping under _mapping
    from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - compatibility for newer onnx
    from onnx import _mapping  # type: ignore[attr-defined]

    NP_TYPE_TO_TENSOR_TYPE = {
        dtype_map.np_dtype: tensor_type
        for tensor_type, dtype_map in _mapping.TENSOR_TYPE_MAP.items()
    }
import torch
from torch.export import export as export_program

from onnx import helper, TensorProto
from onnx2torch.converter import convert
try:
    from tests.utils.common import check_onnx_model
    from tests.utils.common import make_model_from_nodes
except ModuleNotFoundError:  # pragma: no cover - fallback when running module directly
    from ..utils.common import check_onnx_model  # type: ignore[assignment]
    from ..utils.common import make_model_from_nodes  # type: ignore[assignment]


def _test_squeeze(
    input_tensor: np.ndarray,
    axes: Optional[List[int]],
    opset_version: int,
    **kwargs,
) -> None:
    test_inputs: Dict[str, Any] = {"input_tensor": input_tensor}

    if axes is not None and len(axes) > 0:
        if opset_version >= 13:
            test_inputs["axes"] = np.array(axes, dtype=np.int64)
        else:
            kwargs["axes"] = axes

        output_shape = np.squeeze(
            input_tensor, axis=tuple(a for a in axes if input_tensor.shape[a] == 1)
        ).shape
    else:
        output_shape = np.squeeze(input_tensor).shape

    node = onnx.helper.make_node(
        op_type="Squeeze",
        inputs=list(test_inputs),
        outputs=["y"],
        **kwargs,
    )

    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
        opset_version=opset_version,
        outputs_info=(
            make_tensor_value_info(
                name="y",
                elem_type=NP_TYPE_TO_TENSOR_TYPE[input_tensor.dtype],
                shape=output_shape,
            ),
        ),
    )
    check_onnx_model(model, test_inputs)


@pytest.mark.filterwarnings("ignore::torch.jit._trace.TracerWarning")
@pytest.mark.parametrize("opset_version", [11, 13, 21])
@pytest.mark.parametrize(
    "shape, axes",
    (
        ([1, 3, 4, 5], [0]),
        ([1, 3, 1, 5], [-2]),
        ([1, 3, 1, 5], [0, 2]),
        ([1, 3, 1, 5], [2, 0]),
        ([1, 3, 1, 1, 1, 5, 1], [2, 0, 6]),
        ([1, 3, 1, 5], [0, -2]),
        ([1, 3, 1, 5], [-2, 0]),
        ([1, 3, 1, 5], None),
        ([1, 1, 1, 1], None),
        ([1], None),
        ([3, 3, 3], None),
    ),
)
def test_squeeze(  # pylint: disable=missing-function-docstring
    shape: List[int],
    axes: List[int],
    opset_version: int,
) -> None:
    x = np.random.randn(*shape).astype(np.float32)
    _test_squeeze(input_tensor=x, axes=axes, opset_version=opset_version)


@pytest.mark.filterwarnings("ignore::torch.jit._trace.TracerWarning")
@pytest.mark.parametrize("opset_version", [13, 21])
def test_squeeze_axes_initializer(opset_version: int) -> None:  # pylint: disable=missing-function-docstring
    x = np.random.randn(1, 3, 1, 5).astype(np.float32)
    axes = np.array([0, 2], dtype=np.int64)

    node = onnx.helper.make_node(
        op_type="Squeeze",
        inputs=["input_tensor", "axes"],
        outputs=["y"],
    )

    output_shape = np.squeeze(x, axis=(0, 2)).shape
    model = make_model_from_nodes(
        nodes=node,
        initializers={"axes": axes},
        inputs_example={"input_tensor": x},
        outputs_info=(
            make_tensor_value_info(
                name="y",
                elem_type=NP_TYPE_TO_TENSOR_TYPE[x.dtype],
                shape=output_shape,
            ),
        ),
        opset_version=opset_version,
    )

    check_onnx_model(model, {"input_tensor": x})


@pytest.mark.filterwarnings("ignore::torch.jit._trace.TracerWarning")
def test_squeeze_split_transpose_allows_export() -> None:  # pylint: disable=missing-function-docstring
    qkv_info = make_tensor_value_info(
        name="qkv",
        elem_type=TensorProto.FLOAT,
        shape=[1, 12, 3, 128, 64],
    )
    output_info = make_tensor_value_info(
        name="out",
        elem_type=TensorProto.FLOAT,
        shape=None,
    )

    splits = helper.make_tensor("splits", TensorProto.INT64, dims=[3], vals=[1, 1, 1])
    axes = helper.make_tensor("axes", TensorProto.INT64, dims=[1], vals=[2])

    split_node = helper.make_node(
        "Split", inputs=["qkv", "splits"], outputs=["q", "k", "v"], axis=2
    )
    squeeze_node = helper.make_node(
        "Squeeze", inputs=["q", "axes"], outputs=["q_squeezed"]
    )
    transpose_node = helper.make_node(
        "Transpose",
        inputs=["q_squeezed"],
        outputs=["out"],
        perm=[0, 1, 3, 2],
    )

    graph = helper.make_graph(
        nodes=[split_node, squeeze_node, transpose_node],
        name="squeeze-transpose",
        inputs=[qkv_info],
        outputs=[output_info],
        initializer=[splits, axes],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    module = convert(model)
    module.eval()

    sample = torch.zeros(1, 12, 3, 128, 64)
    exported = export_program(module, (sample,))
    assert exported is not None
