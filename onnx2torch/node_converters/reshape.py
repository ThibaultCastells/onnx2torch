__all__ = [
    "OnnxReshape",
]

from typing import List
from typing import Optional
from typing import Tuple

import torch
from torch import nn

try:  # pragma: no cover - older torch versions
    from torch._dynamo import is_compiling as _dynamo_is_compiling
except ImportError:  # pragma: no cover - fallback when dynamo is unavailable

    def _dynamo_is_compiling() -> bool:
        return False


from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_graph import ValueType
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.custom_export_to_onnx import DefaultExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport
from onnx2torch.utils.shape_utils import shape_tensor_to_sequence

try:  # pragma: no cover - FakeTensor may be unavailable on older torch releases
    from torch._subclasses.fake_tensor import FakeTensor
except ImportError:  # pragma: no cover - keep isinstance checks safe
    FakeTensor = ()  # type: ignore[assignment]


class OnnxReshape(nn.Module, OnnxToTorchModuleWithCustomExport):  # pylint: disable=missing-class-docstring
    def __init__(self, static_shape: Optional[Tuple[int, ...]] = None) -> None:
        super().__init__()
        self._static_shape = tuple(static_shape) if static_shape is not None else None

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

        # Unpack one scalar per dimension so PyTorch receives SymInt args when tracing.
        target_dims = shape_tensor_to_sequence(adjusted_shape)

        if not target_dims:
            return input_tensor.reshape(())

        return input_tensor.reshape(*target_dims)

    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_tensor: torch.Tensor,
        shape: torch.Tensor,
    ) -> torch.Tensor:
        def _forward() -> torch.Tensor:
            if self._static_shape is not None:
                return self._reshape_with_static_shape(input_tensor)

            if isinstance(
                shape, FakeTensor
            ):  # pragma: no cover - exercised under torch.export
                raise NotImplementedError(
                    "Dynamic Reshape shapes are not supported in fake tensor mode yet"
                )

            return self._do_reshape(input_tensor, shape)

        if torch.onnx.is_in_onnx_export():
            return DefaultExportToOnnx.export(
                _forward, "Reshape", input_tensor, shape, {}
            )

        return _forward()

    def _reshape_with_static_shape(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self._static_shape is None:
            raise RuntimeError("Static shape is not initialised")

        input_rank = input_tensor.dim()
        dims: List[Optional[int]] = []
        negative_index: Optional[int] = None
        divisor = 1

        for index, raw_dim in enumerate(self._static_shape):
            if raw_dim == 0:
                value = input_tensor.shape[index] if index < input_rank else 1
            elif raw_dim == -1:
                if negative_index is not None:
                    raise ValueError("Reshape with multiple -1 dimensions is invalid")
                value = None
                negative_index = index
            else:
                value = raw_dim

            dims.append(value)
            if value is not None:
                divisor = divisor * value

        if negative_index is not None:
            inferred = input_tensor.numel() // divisor
            dims[negative_index] = inferred

        if any(dimension is None for dimension in dims):
            raise ValueError("Failed to resolve all dimensions for static reshape")

        resolved_dims = tuple(dimension for dimension in dims if dimension is not None)
        if not resolved_dims:
            return input_tensor.reshape(())

        return input_tensor.reshape(*resolved_dims)


@add_converter(operation_type="Reshape", version=5)
@add_converter(operation_type="Reshape", version=13)
@add_converter(operation_type="Reshape", version=14)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    if node.attributes.get("allowzero", 0) == 1:
        raise NotImplementedError('"allowzero=1" is not implemented')

    static_shape: Optional[Tuple[int, ...]] = None
    shape_input_name = node.input_values[1]
    value_type = graph.value_type(shape_input_name)

    if value_type == ValueType.GRAPH_INITIALIZER:
        static_shape = tuple(
            int(dim) for dim in graph.initializers[shape_input_name].to_numpy().tolist()
        )
    elif value_type == ValueType.NODE_OUTPUT:
        producer_node, _ = graph.value_as_node_output(shape_input_name)
        if producer_node.operation_type == "Constant":
            constant_value = producer_node.attributes.get("value")
            if constant_value is not None:
                static_shape = tuple(
                    int(dim) for dim in constant_value.to_numpy().tolist()
                )

    return OperationConverterResult(
        torch_module=OnnxReshape(static_shape=static_shape),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
