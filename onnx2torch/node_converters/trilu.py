__all__ = [
    "OnnxTrilu",
]

from typing import Optional
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

try:  # pragma: no cover - FakeTensor not available on older torch versions
    from torch._subclasses.fake_tensor import FakeTensor
except ImportError:  # pragma: no cover - keep isinstance checks safe
    FakeTensor = ()  # type: ignore[assignment]


class OnnxTrilu(nn.Module, OnnxToTorchModuleWithCustomExport):
    """Implementation of the ONNX Trilu operator."""

    def __init__(self, upper: bool) -> None:
        super().__init__()
        self._upper = upper

    @staticmethod
    def _as_scalar_tensor(
        value: Optional[Union[torch.Tensor, float, int]],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        device = reference.device
        dtype = torch.int64

        if value is None:
            return torch.zeros((), dtype=dtype, device=device)

        if isinstance(value, torch.Tensor):
            tensor = value.to(dtype=dtype)
            if tensor.dim() == 0:
                if tensor.device != device:
                    tensor = tensor.to(device=device)
                return tensor

            if tensor.numel() == 1:
                tensor = torch.ops.aten.reshape.default(tensor, [])
                if tensor.device != device:
                    tensor = tensor.to(device=device)
                return tensor

            raise ValueError("Trilu diagonal input must contain a single value")

        return torch.ops.aten.scalar_tensor.default(
            int(value),
            dtype=dtype,
            device=device,
        )

    def _make_mask(
        self,
        input_tensor: torch.Tensor,
        diagonal: torch.Tensor,
    ) -> torch.Tensor:
        rank = input_tensor.dim()
        if rank < 2:
            return torch.ones_like(input_tensor, dtype=torch.bool)

        diagonal = diagonal.to(dtype=torch.int64)
        if diagonal.device != input_tensor.device:
            diagonal = diagonal.to(device=input_tensor.device)

        row_count = torch.ops.aten.sym_size.int(input_tensor, rank - 2)
        col_count = torch.ops.aten.sym_size.int(input_tensor, rank - 1)

        row_indices = torch.arange(
            row_count,
            dtype=torch.int64,
            device=input_tensor.device,
        )
        col_indices = torch.arange(
            col_count,
            dtype=torch.int64,
            device=input_tensor.device,
        )

        row_indices = torch.ops.aten.unsqueeze.default(row_indices, -1)
        col_indices = torch.ops.aten.unsqueeze.default(col_indices, 0)
        diff = torch.ops.aten.sub.Tensor(col_indices, row_indices)

        if self._upper:
            mask = torch.ops.aten.ge.Tensor(diff, diagonal)
        else:
            mask = torch.ops.aten.le.Tensor(diff, diagonal)

        for _ in range(rank - 2):
            mask = torch.ops.aten.unsqueeze.default(mask, 0)

        return mask.to(dtype=torch.bool)

    def _apply_trilu(
        self,
        input_tensor: torch.Tensor,
        diagonal: torch.Tensor,
    ) -> torch.Tensor:
        mask = self._make_mask(input_tensor, diagonal)
        if isinstance(input_tensor, FakeTensor):
            # Fake tensors cannot mix devices, keep zeros on the same fake backend.
            zeros = torch.ops.aten.zeros_like.default(input_tensor)
        else:
            zeros = torch.zeros_like(input_tensor)

        return torch.where(mask, input_tensor, zeros)

    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_tensor: torch.Tensor,
        k: Optional[Union[torch.Tensor, float, int]] = None,
    ) -> torch.Tensor:
        def _forward() -> torch.Tensor:
            diagonal = self._as_scalar_tensor(k, reference=input_tensor)
            return self._apply_trilu(input_tensor, diagonal)

        if torch.onnx.is_in_onnx_export():
            args = (input_tensor,) if k is None else (input_tensor, k)
            return DefaultExportToOnnx.export(
                _forward,
                "Trilu",
                *args,
                {"upper_i": int(self._upper)},
            )

        return _forward()


@add_converter(operation_type="Trilu", version=14)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    del graph
    upper = bool(node.attributes.get("upper", 0))

    return OperationConverterResult(
        torch_module=OnnxTrilu(upper=upper),
        onnx_mapping=onnx_mapping_from_node(node),
    )
