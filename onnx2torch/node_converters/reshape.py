__all__ = [
    "OnnxReshape",
]

from typing import List
from typing import Optional
from typing import Tuple

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_graph import ValueType
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.custom_export_to_onnx import DefaultExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport
from onnx2torch.utils.shape_utils import shape_tensor_to_sequence


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

        dims = [
            torch.ops.aten.scalar_tensor.default(
                torch.ops.aten.sym_size.int(input_tensor, index),
                dtype=torch.int64,
                device=device,
            )
            for index in range(input_tensor.dim())
        ]

        if dims:
            shape_tensor = torch.ops.aten.stack.default(dims, 0)
        else:
            shape_tensor = torch.zeros((0,), dtype=torch.int64, device=device)

        current_length = int(shape_tensor.numel())
        if current_length == target_length:
            return shape_tensor.to(dtype=dtype)

        if current_length > target_length:
            trimmed = torch.ops.aten.slice.Tensor(shape_tensor, 0, 0, target_length, 1)
            return trimmed.to(dtype=dtype)

        pad_length = target_length - current_length
        padding = torch.ones((pad_length,), dtype=shape_tensor.dtype, device=device)
        padded = torch.cat((shape_tensor, padding))
        return padded.to(dtype=dtype)

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

        if target_dims:
            dims_list = list(target_dims)
            if all(isinstance(dim, int) for dim in dims_list):
                product = 1
                for dim in dims_list:
                    product *= dim

                input_numel = input_tensor.numel()
                try:
                    input_numel_int = int(input_numel)
                except Exception:
                    input_numel_int = -1

                if input_numel_int >= 0 and product == input_numel_int:
                    target_dims = tuple(dims_list)
                else:
                    largest_index = max(
                        range(len(dims_list)),
                        key=lambda index: abs(dims_list[index]),
                    )
                    dims_list[largest_index] = -1
                    target_dims = tuple(dims_list)

        if not target_dims:
            return torch.ops.aten.view.default(input_tensor, [])

        return torch.reshape(input_tensor, list(target_dims))

    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_tensor: torch.Tensor,
        shape: torch.Tensor,
    ) -> torch.Tensor:
        def _forward() -> torch.Tensor:
            if self._static_shape is not None:
                return self._reshape_with_static_shape(input_tensor)

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
            return torch.ops.aten.view.default(input_tensor, [])

        product = 1
        for dimension in resolved_dims:
            product *= dimension

        input_numel = input_tensor.numel()
        try:
            input_numel_int = int(input_numel)
        except Exception:
            input_numel_int = None

        if input_numel_int is None:
            shape_tensor = torch.tensor(self._static_shape, dtype=torch.int64)
            try:
                shape_tensor = shape_tensor.to(device=input_tensor.device)
            except (RuntimeError, TypeError):
                pass
            return self._do_reshape(input_tensor, shape_tensor)

        if product != input_numel_int:
            dims_list = list(resolved_dims)
            largest_index = max(
                range(len(dims_list)), key=lambda index: abs(dims_list[index])
            )
            candidate_value = dims_list[largest_index]
            if candidate_value == 0:
                shape_tensor = torch.tensor(self._static_shape, dtype=torch.int64)
                try:
                    shape_tensor = shape_tensor.to(device=input_tensor.device)
                except (RuntimeError, TypeError):
                    pass
                return self._do_reshape(input_tensor, shape_tensor)

            divisor = product // candidate_value
            if divisor == 0 or input_numel_int % divisor != 0:
                shape_tensor = torch.tensor(self._static_shape, dtype=torch.int64)
                try:
                    shape_tensor = shape_tensor.to(device=input_tensor.device)
                except (RuntimeError, TypeError):
                    pass
                return self._do_reshape(input_tensor, shape_tensor)

            inferred_dim = input_numel_int // divisor
            if inferred_dim <= 0:
                shape_tensor = torch.tensor(self._static_shape, dtype=torch.int64)
                try:
                    shape_tensor = shape_tensor.to(device=input_tensor.device)
                except (RuntimeError, TypeError):
                    pass
                return self._do_reshape(input_tensor, shape_tensor)

            dims_list[largest_index] = inferred_dim
            resolved_dims = tuple(dims_list)

            product = 1
            for dimension in resolved_dims:
                product *= dimension

            if product != input_numel_int:
                shape_tensor = torch.tensor(self._static_shape, dtype=torch.int64)
                try:
                    shape_tensor = shape_tensor.to(device=input_tensor.device)
                except (RuntimeError, TypeError):
                    pass
                return self._do_reshape(input_tensor, shape_tensor)

        return torch.ops.aten.view.default(input_tensor, list(resolved_dims))


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
