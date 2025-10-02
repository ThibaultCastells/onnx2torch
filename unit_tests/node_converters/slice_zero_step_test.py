import pytest
import torch

from onnx2torch.node_converters.slice import OnnxSlice


def test_slice_static_zero_step_defaults_to_one() -> None:  # pylint: disable=missing-function-docstring
    module = OnnxSlice()

    input_tensor = torch.arange(6, dtype=torch.float32)
    starts = torch.tensor([0], dtype=torch.int64)
    ends = torch.tensor([6], dtype=torch.int64)
    axes = torch.tensor([0], dtype=torch.int64)
    steps = torch.tensor([0], dtype=torch.int64)

    with pytest.warns(RuntimeWarning, match="Slice step of 0 encountered"):
        result = module(input_tensor, starts, ends, axes=axes, steps=steps)

    torch.testing.assert_close(result, input_tensor)
