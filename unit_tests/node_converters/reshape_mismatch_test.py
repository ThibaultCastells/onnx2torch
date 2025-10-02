import torch

from onnx2torch.node_converters.reshape import OnnxReshape


def test_reshape_adjusts_mismatched_static_dimension() -> None:
    """Reshape should adapt static dimensions when element counts disagree."""

    module = OnnxReshape()

    input_tensor = torch.randn(1, 4096 * 16 * 128)
    target_shape = torch.tensor([1, 32768, 16, 128], dtype=torch.int64)

    result = module(input_tensor, target_shape)

    assert result.shape == (1, 4096, 16, 128)


def test_reshape_static_shape_adjusts_mismatched_dimension() -> None:
    """Static reshape should clamp oversized constants to the actual sequence length."""

    module = OnnxReshape(static_shape=(1, 32768, 16, 128))

    input_tensor = torch.randn(1, 4096, 2048)
    dummy_shape = torch.tensor([], dtype=torch.int64)

    result = module(input_tensor, dummy_shape)

    assert result.shape == (1, 4096, 16, 128)
