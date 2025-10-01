from __future__ import annotations

import numpy as np
import pytest
import torch
from onnx import TensorProto
from onnx import helper
from onnx import numpy_helper

from onnx2torch import convert


def _invalid_reshape_model():
    input_value = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2])
    output_value = helper.make_tensor_value_info("output", TensorProto.FLOAT, [3])

    shape_initializer = numpy_helper.from_array(
        np.array([3], dtype=np.int64), name="shape"
    )

    reshape_node = helper.make_node(
        "Reshape",
        inputs=["input", "shape"],
        outputs=["output"],
        name="bad_reshape",
    )

    graph = helper.make_graph(
        [reshape_node],
        "reshape_failure",
        [input_value],
        [output_value],
        initializer=[shape_initializer],
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 14)],
    )
    return model


def test_runtime_errors_include_onnx_context():
    module = convert(_invalid_reshape_model())

    with pytest.raises(RuntimeError) as exc_info:
        module(torch.randn(2))

    notes = getattr(exc_info.value, "__notes__", None) or []
    joined_notes = "\n".join(notes)

    assert "op_type: Reshape" in joined_notes
    assert "node_name: bad_reshape" in joined_notes
    assert "inputs: input, shape" in joined_notes
    assert "outputs: output" in joined_notes
