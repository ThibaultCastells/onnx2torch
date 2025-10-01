__all__ = [
    "OnnxReshape",
]

import torch
from torch import nn

try:  # pragma: no cover - older torch versions
    from torch._dynamo import is_compiling as _dynamo_is_compiling
except ImportError:  # pragma: no cover - fallback when dynamo is unavailable

    def _dynamo_is_compiling() -> bool:
        return False


from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.custom_export_to_onnx import DefaultExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport


class OnnxReshape(nn.Module, OnnxToTorchModuleWithCustomExport):  # pylint: disable=missing-class-docstring
    @staticmethod
    def _input_shape_tensor(
        input_tensor: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
        target_length: int,
    ) -> torch.Tensor:
        if target_length == 0:
            return torch.zeros((0,), dtype=dtype, device=device)

        input_rank = input_tensor.dim()

        if _dynamo_is_compiling():
            parts = []
            for index in range(target_length):
                if index < input_rank:
                    dim_value = torch.ops.aten.sym_size.int(input_tensor, index)
                else:
                    dim_value = 1
                parts.append(
                    torch.ops.aten.scalar_tensor.default(
                        dim_value,
                        dtype=dtype,
                        device=device,
                    )
                )
            return torch.stack(parts)

        base_dims = list(input_tensor.shape)
        if target_length <= input_rank:
            dims = base_dims[:target_length]
        else:
            dims = base_dims + [1] * (target_length - input_rank)

        return torch.tensor(dims, dtype=dtype, device=device)

    @classmethod
    def _do_reshape(
        cls, input_tensor: torch.Tensor, shape: torch.Tensor
    ) -> torch.Tensor:
        shape_tensor = shape.to(dtype=torch.int64)
        input_shape_tensor = cls._input_shape_tensor(
            input_tensor,
            shape_tensor.dtype,
            shape_tensor.device,
            shape_tensor.numel(),
        )
        adjusted_shape = torch.where(
            shape_tensor == 0, input_shape_tensor, shape_tensor
        )
        return torch.ops.aten.reshape.default(input_tensor, adjusted_shape)

    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_tensor: torch.Tensor,
        shape: torch.Tensor,
    ) -> torch.Tensor:
        def _forward() -> torch.Tensor:
            return self._do_reshape(input_tensor, shape)

        if torch.onnx.is_in_onnx_export():
            return DefaultExportToOnnx.export(
                _forward, "Reshape", input_tensor, shape, {}
            )

        return _forward()


@add_converter(operation_type="Reshape", version=5)
@add_converter(operation_type="Reshape", version=13)
@add_converter(operation_type="Reshape", version=14)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    if node.attributes.get("allowzero", 0) == 1:
        raise NotImplementedError('"allowzero=1" is not implemented')

    return OperationConverterResult(
        torch_module=OnnxReshape(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
