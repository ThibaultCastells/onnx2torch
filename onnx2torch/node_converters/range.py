# pylint: disable=missing-docstring
__all__ = [
    "OnnxRange",
]

from typing import Union

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.custom_export_to_onnx import DefaultExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport


class OnnxRange(nn.Module, OnnxToTorchModuleWithCustomExport):
    def __init__(self):
        super().__init__()
        self.register_buffer("dummy_buffer", torch.Tensor(), persistent=False)

    @staticmethod
    def _infer_device(
        start: Union[torch.Tensor, float, int],
        limit: Union[torch.Tensor, float, int],
        delta: Union[torch.Tensor, float, int],
        fallback: torch.device,
    ) -> torch.device:
        for value in (start, limit, delta):
            if isinstance(value, torch.Tensor):
                return value.device
        return fallback

    @staticmethod
    def _as_tensor(
        value: Union[torch.Tensor, float, int], device: torch.device
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=device)
        return torch.tensor(value, device=device)

    @staticmethod
    def _promote_dtype(
        start: torch.Tensor,
        limit: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.dtype:
        dtype = torch.promote_types(start.dtype, limit.dtype)
        dtype = torch.promote_types(dtype, delta.dtype)
        return dtype

    @staticmethod
    def _calc_dtype(base_dtype: torch.dtype) -> torch.dtype:
        if base_dtype.is_floating_point:
            return base_dtype
        # Use float64 to minimise precision loss for large integer ranges.
        return torch.float64

    def _arange(
        self,
        start: Union[torch.Tensor, float, int],
        limit: Union[torch.Tensor, float, int],
        delta: Union[torch.Tensor, float, int],
    ) -> torch.Tensor:
        device = self._infer_device(start, limit, delta, self.dummy_buffer.device)

        start_tensor = self._as_tensor(start, device)
        limit_tensor = self._as_tensor(limit, device)
        delta_tensor = self._as_tensor(delta, device)

        base_dtype = self._promote_dtype(start_tensor, limit_tensor, delta_tensor)
        start_tensor = start_tensor.to(base_dtype)
        limit_tensor = limit_tensor.to(base_dtype)
        delta_tensor = delta_tensor.to(base_dtype)

        calc_dtype = self._calc_dtype(base_dtype)
        start_calc = start_tensor.to(calc_dtype)
        limit_calc = limit_tensor.to(calc_dtype)
        delta_calc = delta_tensor.to(calc_dtype)

        steps = torch.ceil((limit_calc - start_calc) / delta_calc)
        steps = torch.clamp_min(steps, 0)
        steps_int = steps.to(torch.int64)

        seq = torch.arange(steps_int, device=device, dtype=calc_dtype)
        values = start_calc + delta_calc * seq
        return values.to(base_dtype)

    def forward(
        self,
        start: Union[torch.Tensor, float, int],
        limit: Union[torch.Tensor, float, int],
        delta: Union[torch.Tensor, float, int],
    ) -> torch.Tensor:
        def _forward() -> torch.Tensor:
            return self._arange(start, limit, delta)

        if torch.onnx.is_in_onnx_export():
            return DefaultExportToOnnx.export(
                _forward, "Range", start, limit, delta, {}
            )

        return _forward()


@add_converter(operation_type="Range", version=11)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    del graph
    return OperationConverterResult(
        torch_module=OnnxRange(),
        onnx_mapping=onnx_mapping_from_node(node),
    )
