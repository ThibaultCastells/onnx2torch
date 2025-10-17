__all__ = [
    "OnnxCumSum",
]
from typing import Optional

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.common import get_const_value
from onnx2torch.utils.shape_utils import sequence_to_symint_tuple


def _arbitrary_dim_shift_and_insert_zero(
    input_tensor: torch.Tensor,
    insert_dim: int,
) -> torch.Tensor:
    # single item shift
    slice_index, insertion = [[slice(None)] * len(input_tensor.shape)] * 2
    insert_dim_size = input_tensor.shape[insert_dim]

    slice_index[insert_dim] = slice(0, -1)
    slice_index = tuple(slice_index)
    tensor_slice = input_tensor[slice_index]

    insert_index = torch.arange(
        start=1, end=insert_dim_size, dtype=torch.int64, device=input_tensor.device
    )
    index_shape = [1] * len(input_tensor.shape)
    index_shape[insert_dim] = insert_dim_size - 1

    insert_index = torch.reshape(insert_index, index_shape)
    insert_index = insert_index + torch.zeros_like(
        tensor_slice, dtype=torch.int64, device=input_tensor.device
    )

    input_tensor = torch.scatter(
        input=input_tensor,
        dim=insert_dim,
        index=insert_index,
        src=tensor_slice,
    )

    insertion[insert_dim] = slice(0, 1)
    insertion = tuple(insertion)
    input_tensor[insertion] = 0

    return input_tensor


class OnnxCumSum(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(
        self,
        exclusive: bool = False,
        reverse: bool = False,
        static_axis: Optional[int] = None,
    ):
        super().__init__()
        self.exclusive = exclusive
        self.reverse = reverse
        self._static_axis = static_axis

    @staticmethod
    def _normalise_axis(axis: int, rank: int) -> int:
        if rank == 0:
            return 0

        normalised = axis if axis >= 0 else axis + rank
        if normalised < 0 or normalised >= rank:
            raise IndexError(
                f"Axis {axis} is out of range for tensor rank {rank} in CumSum."
            )

        return normalised

    def _resolve_axis(
        self,
        input_tensor: torch.Tensor,
        axis: Optional[torch.Tensor],
    ) -> int:
        if self._static_axis is not None:
            return self._normalise_axis(self._static_axis, input_tensor.dim())

        if axis is None:
            raise RuntimeError("CumSum requires the axis input to be provided.")

        axis_values = sequence_to_symint_tuple(axis)
        if not axis_values:
            resolved_axis = 0
        else:
            resolved_axis = axis_values[0]

        try:
            axis_int = int(resolved_axis)
        except Exception as error:  # noqa: BLE001 - surface actionable failure
            raise NotImplementedError(
                "CumSum currently requires integer axis values."
            ) from error

        return self._normalise_axis(axis_int, input_tensor.dim())

    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_tensor: torch.Tensor,
        axis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        axis_index = self._resolve_axis(input_tensor, axis)
        if self.reverse:
            input_tensor = torch.flip(input_tensor, dims=(axis_index,))

        if self.exclusive:
            input_tensor = _arbitrary_dim_shift_and_insert_zero(
                input_tensor, insert_dim=axis_index
            )

        input_tensor = torch.cumsum(input_tensor, dim=axis_index)

        if self.reverse:
            input_tensor = torch.flip(input_tensor, dims=(axis_index,))

        return input_tensor


@add_converter(operation_type="CumSum", version=11)
@add_converter(operation_type="CumSum", version=14)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    node_attributes = node.attributes
    exclusive = bool(node_attributes.get("exclusive", 0))
    reverse = bool(node_attributes.get("reverse", 0))

    static_axis: Optional[int] = None
    if len(node.input_values) > 1 and node.input_values[1]:
        axis_input_name = node.input_values[1]
        try:
            axis_value = get_const_value(axis_input_name, graph)
        except KeyError:
            static_axis = None
        else:
            if isinstance(axis_value, torch.Tensor):
                if axis_value.numel() != 1:
                    raise NotImplementedError(
                        "CumSum supports only scalar axis tensors."
                    )
                static_axis = int(axis_value.reshape(-1)[0].item())
            else:
                static_axis = int(axis_value)

    return OperationConverterResult(
        torch_module=OnnxCumSum(exclusive, reverse, static_axis=static_axis),
        onnx_mapping=onnx_mapping_from_node(node),
    )
