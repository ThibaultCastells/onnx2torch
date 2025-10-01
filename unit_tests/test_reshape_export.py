from __future__ import annotations

import torch

from onnx2torch.node_converters.reshape import OnnxReshape


def test_reshape_supports_export_with_zero_copy_dims():
    module = OnnxReshape(static_shape=[0, -1])

    input_tensor = torch.randn(2, 3, 5)
    # ONNX semantics: 0 copies the corresponding dimension from the input.
    shape = torch.tensor([0, -1], dtype=torch.int64)

    exported = torch.export.export(module, (input_tensor, shape))

    result = exported.module()(input_tensor, shape)
    assert result.shape == (2, 15)
