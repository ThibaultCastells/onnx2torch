__all__ = [
    "OnnxSqueezeStaticAxes",
    "OnnxSqueezeDynamicAxes",
]

from typing import List
from typing import Optional

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_graph import ValueType
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxMapping
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import get_onnx_version
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.custom_export_to_onnx import DefaultExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport


class OnnxSqueezeStaticAxes(nn.Module, OnnxToTorchModuleWithCustomExport):  # pylint: disable=missing-class-docstring
    def __init__(self, axes: Optional[List[int]] = None):
        super().__init__()
        if axes is not None:
            axes = sorted(axes, reverse=True)

        self.axes = axes

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        def _forward():
            if not self.axes:
                return torch.squeeze(input_tensor)

            result = input_tensor
            for axes_id in self.axes:
                result = torch.squeeze(result, dim=axes_id)

            return result

        if torch.onnx.is_in_onnx_export() and get_onnx_version() >= 13:
            args = [input_tensor]
            if self.axes:
                axes = torch.tensor(
                    self.axes, device=input_tensor.device, dtype=torch.int64
                )
                args.append(axes)

            return DefaultExportToOnnx.export(_forward, "Squeeze", *args, {})

        return _forward()


class OnnxSqueezeDynamicAxes(  # pylint: disable=missing-class-docstring
    nn.Module,
    OnnxToTorchModuleWithCustomExport,
):
    @staticmethod
    def is_empty_axes(axes: torch.Tensor) -> bool:  # pylint: disable=missing-function-docstring
        return axes is None or axes.nelement() == 0

    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_tensor: torch.Tensor,
        axes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        def _forward():
            if OnnxSqueezeDynamicAxes.is_empty_axes(axes):
                return torch.squeeze(input_tensor)

            result = input_tensor
            for axes_id in torch.sort(axes, descending=True).values:
                result = torch.squeeze(result, dim=axes_id)

            return result

        if torch.onnx.is_in_onnx_export():
            args = [input_tensor]
            if not self.is_empty_axes(axes):
                args.append(axes)

            return DefaultExportToOnnx.export(_forward, "Squeeze", *args, {})

        return _forward()


@add_converter(operation_type="Squeeze", version=1)
@add_converter(operation_type="Squeeze", version=11)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    axes = node.attributes.get("axes", None)
    return OperationConverterResult(
        torch_module=OnnxSqueezeStaticAxes(axes=axes),
        onnx_mapping=onnx_mapping_from_node(node),
    )


@add_converter(operation_type="Squeeze", version=13)
@add_converter(operation_type="Squeeze", version=21)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    axes = _maybe_get_static_axes(node=node, graph=graph)
    if axes is not None:
        mapping_inputs = (node.input_values[0],)
        return OperationConverterResult(
            torch_module=OnnxSqueezeStaticAxes(axes=axes),
            onnx_mapping=OnnxMapping(
                inputs=mapping_inputs,
                outputs=node.output_values,
            ),
        )

    return OperationConverterResult(
        torch_module=OnnxSqueezeDynamicAxes(),
        onnx_mapping=onnx_mapping_from_node(node),
    )


def _maybe_get_static_axes(  # pylint: disable=too-many-return-statements
    node: OnnxNode,
    graph: OnnxGraph,
) -> Optional[List[int]]:
    if len(node.input_values) < 2:
        return None

    axes_input_name = node.input_values[1]
    value_type = graph.value_type(axes_input_name)

    if value_type == ValueType.GRAPH_INITIALIZER:
        axes_array = graph.initializers[axes_input_name].to_numpy().reshape(-1)
        axes_list = [int(axis) for axis in axes_array.tolist()]
        return axes_list or None

    if value_type == ValueType.NODE_OUTPUT:
        producer_node, _ = graph.value_as_node_output(axes_input_name)
        if producer_node.operation_type == "Constant":
            constant_value = producer_node.attributes.get("value")
            if constant_value is not None:
                axes_array = constant_value.to_numpy().reshape(-1)
                axes_list = [int(axis) for axis in axes_array.tolist()]
                return axes_list or None

    return None
