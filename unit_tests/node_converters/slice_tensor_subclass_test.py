import torch

from onnx2torch.node_converters.slice import OnnxSlice


class NumpyGuardTensor(torch.Tensor):
    """Tensor subclass that forbids numpy conversion to mirror FakeTensor behaviour."""

    @staticmethod
    def __new__(cls, elem: torch.Tensor):  # type: ignore[override]
        return torch.Tensor._make_subclass(
            cls, elem, elem.requires_grad
        )  # pragma: no cover - thin wrapper

    def numpy(self):  # pragma: no cover - behaviour under test
        raise RuntimeError(".numpy() is not supported for tensor subclasses.")


def test_slice_accepts_tensor_subclasses_without_numpy() -> None:  # pylint: disable=missing-function-docstring
    module = OnnxSlice()

    input_tensor = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
    starts = NumpyGuardTensor(torch.tensor([0, 0, 1], dtype=torch.int64))
    ends = NumpyGuardTensor(torch.tensor([2, 2, 3], dtype=torch.int64))
    axes = NumpyGuardTensor(torch.tensor([0, 1, -1], dtype=torch.int64))
    steps = NumpyGuardTensor(torch.tensor([1, 1, 1], dtype=torch.int64))

    output = module(input_tensor, starts, ends, axes, steps)
    expected = input_tensor[:, :, 1:3]

    assert torch.allclose(output, expected)
