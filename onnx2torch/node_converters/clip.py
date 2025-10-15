__all__ = [
    "OnnxClip",
]

from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import nn
from torch.types import Number

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxMapping
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import get_const_value
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxClip(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(
        self,
        min_val: Optional[Union[torch.Tensor, Number]] = None,
        max_val: Optional[Union[torch.Tensor, Number]] = None,
        dynamic_min: bool = False,
        dynamic_max: bool = False,
    ):
        super().__init__()
        self.dynamic_min = dynamic_min
        self.dynamic_max = dynamic_max

        min_tensor = min_val if isinstance(min_val, torch.Tensor) else None
        max_tensor = max_val if isinstance(max_val, torch.Tensor) else None

        self.register_buffer("min_tensor", min_tensor, persistent=False)
        self.register_buffer("max_tensor", max_tensor, persistent=False)

        self.min_scalar = None if isinstance(min_val, torch.Tensor) else min_val
        self.max_scalar = None if isinstance(max_val, torch.Tensor) else max_val

    @staticmethod
    def _resolve_bound(
        scalar_value: Optional[Number],
        tensor_value: Optional[torch.Tensor],
        dynamic_value: Optional[torch.Tensor],
        reference: torch.Tensor,
    ) -> Optional[Union[torch.Tensor, Number]]:
        if dynamic_value is not None:
            return dynamic_value.to(dtype=reference.dtype, device=reference.device)

        if tensor_value is not None:
            if (
                tensor_value.dtype != reference.dtype
                or tensor_value.device != reference.device
            ):
                return tensor_value.to(
                    dtype=reference.dtype,
                    device=reference.device,
                )
            return tensor_value

        return scalar_value

    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_tensor: torch.Tensor,
        *extra_inputs: torch.Tensor,
    ) -> torch.Tensor:
        extra_iter = iter(extra_inputs)
        dynamic_min = next(extra_iter, None) if self.dynamic_min else None
        dynamic_max = next(extra_iter, None) if self.dynamic_max else None

        min_value = self._resolve_bound(
            scalar_value=self.min_scalar,
            tensor_value=self.min_tensor,
            dynamic_value=dynamic_min,
            reference=input_tensor,
        )
        max_value = self._resolve_bound(
            scalar_value=self.max_scalar,
            tensor_value=self.max_tensor,
            dynamic_value=dynamic_max,
            reference=input_tensor,
        )

        clamp_kwargs = {}
        if min_value is not None:
            clamp_kwargs["min"] = min_value
        if max_value is not None:
            clamp_kwargs["max"] = max_value

        if not clamp_kwargs:
            return input_tensor

        return torch.clamp(input_tensor, **clamp_kwargs)


def _normalize_bound(
    value: Union[torch.Tensor, Number, None]
) -> Optional[Union[torch.Tensor, Number]]:
    if value is None:
        return None

    if isinstance(value, torch.Tensor) and value.ndim == 0:
        return value.item()

    return value


def _create_torch_module(
    min_val: Optional[Union[torch.Tensor, Number]],
    max_val: Optional[Union[torch.Tensor, Number]],
    dynamic_min: bool,
    dynamic_max: bool,
) -> nn.Module:
    if not dynamic_min and not dynamic_max:
        if min_val is None and max_val is None:
            return nn.Identity()

        if isinstance(min_val, Number) and min_val == 0 and max_val is None:
            return nn.ReLU()

        if (
            isinstance(min_val, Number)
            and min_val == 0
            and isinstance(max_val, Number)
            and max_val == 6
        ):
            return nn.ReLU6()

    return OnnxClip(
        min_val=min_val,
        max_val=max_val,
        dynamic_min=dynamic_min,
        dynamic_max=dynamic_max,
    )


def _extract_bound(
    name: Optional[str], graph: OnnxGraph
) -> Tuple[Optional[Union[torch.Tensor, Number]], bool]:
    if not name:
        return None, False

    try:
        value = get_const_value(name, graph)
    except KeyError:
        return None, True

    return _normalize_bound(value), False


@add_converter(operation_type="Clip", version=11)
@add_converter(operation_type="Clip", version=12)
@add_converter(operation_type="Clip", version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    # Min and Max inputs are optional
    min_name = node.input_values[1] if len(node.input_values) > 1 else None
    max_name = node.input_values[2] if len(node.input_values) > 2 else None
    min_val, dynamic_min = _extract_bound(min_name, graph)
    max_val, dynamic_max = _extract_bound(max_name, graph)

    torch_module = _create_torch_module(
        min_val=min_val,
        max_val=max_val,
        dynamic_min=dynamic_min,
        dynamic_max=dynamic_max,
    )

    input_values = [node.input_values[0]]
    if dynamic_min and min_name is not None:
        input_values.append(min_name)
    if dynamic_max and max_name is not None:
        input_values.append(max_name)

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=OnnxMapping(
            inputs=tuple(input_values),
            outputs=node.output_values,
        ),
    )


@add_converter(operation_type="Clip", version=6)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    node_attributes = node.attributes
    min_val = node_attributes.get("min", None)
    max_val = node_attributes.get("max", None)

    torch_module = _create_torch_module(
        min_val=min_val,
        max_val=max_val,
        dynamic_min=False,
        dynamic_max=False,
    )

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node),
    )
