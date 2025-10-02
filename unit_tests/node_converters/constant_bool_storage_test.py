import torch

from onnx2torch.node_converters.constant import OnnxConstant
from onnx2torch.node_converters.constant_of_shape import OnnxConstantOfShape


def test_constant_bool_buffer_stores_as_uint8():
    original = torch.tensor([[True, False]], dtype=torch.bool)
    module = OnnxConstant(original)

    buffers = dict(module.named_buffers())
    stored = buffers["_value_tensor"]

    assert stored.dtype is torch.uint8
    result = module()
    assert result.dtype is torch.bool
    assert torch.equal(result, original)


def test_constant_of_shape_bool_buffer_stores_as_uint8():
    module = OnnxConstantOfShape(torch.tensor(True))

    buffers = dict(module.named_buffers())
    stored = buffers["_value_tensor"]

    assert stored.dtype is torch.uint8
    result = module(torch.tensor([2, 1]))
    assert result.dtype is torch.bool
    assert torch.equal(result, torch.ones((2, 1), dtype=torch.bool))
