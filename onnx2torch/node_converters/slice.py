"""ONNX Slice converter with symbolic-shape friendly semantics."""

__all__ = [
    "OnnxSlice",
]

from typing import List
from typing import Optional
from typing import Tuple
import warnings

import numpy as np
import torch
from torch import nn

try:  # pragma: no cover - SymInt introduced in newer torch versions
    from torch import SymInt  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fall back when SymInt is missing
    SymInt = int  # type: ignore[assignment]

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_graph import ValueType
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.onnx_tensor import OnnxTensor
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.custom_export_to_onnx import DefaultExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport
from onnx2torch.utils.shape_utils import ShapeDimension
from onnx2torch.utils.shape_utils import sequence_to_symint_tuple
from onnx2torch.utils.shape_warmup import is_shape_warmup_active


def _is_symbolic(value: ShapeDimension) -> bool:
    return isinstance(value, SymInt) and not isinstance(value, int)


def _all_concrete(values: Tuple[ShapeDimension, ...]) -> bool:
    return all(not _is_symbolic(value) for value in values)


def _coerce_int_tuple(
    values: Tuple[ShapeDimension, ...],
    *,
    description: str,
) -> Tuple[int, ...]:
    integers: List[int] = []
    for value in values:
        if _is_symbolic(value):
            raise NotImplementedError(
                f"Slice {description} requires concrete integers; received a symbolic value."
            )
        integers.append(int(value))
    return tuple(integers)


def _materialise_static_slice(
    starts: Tuple[int, ...],
    ends: Tuple[int, ...],
    axes: Tuple[int, ...],
    steps: Tuple[int, ...],
) -> Tuple[List[int], List[slice], List[slice]]:
    slices: dict[int, slice] = {}
    flip_dims: List[int] = []

    for start, end, axis, step in zip(starts, ends, axes, steps):
        if step == 0:
            if not is_shape_warmup_active():
                warnings.warn(
                    "Slice step of 0 encountered; treating as 1 for compatibility.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            step = 1
        if step < 0:
            flip_dims.append(axis)
            start, end, step = -start - 1, -end - 1, -step
        slices[axis] = slice(start, end, step)

    if not axes:
        return flip_dims, [], []

    max_axis = max(axes)
    min_axis = min(axes)

    pos_axes = [slices.get(axis, slice(None)) for axis in range(max_axis + 1)]
    neg_axes = [slices.get(axis, slice(None)) for axis in range(min_axis, 0)]

    if neg_axes:
        neg_axes = [Ellipsis] + neg_axes

    return flip_dims, pos_axes, neg_axes


def _execute_static_slice(
    input_tensor: torch.Tensor,
    flip_dims: List[int],
    pos_axes_slices: List[slice],
    neg_axes_slices: List[slice],
) -> torch.Tensor:
    result = input_tensor
    if flip_dims:
        result = torch.flip(result, dims=flip_dims)
    if pos_axes_slices:
        result = result[tuple(pos_axes_slices)]
    if neg_axes_slices:
        result = result[tuple(neg_axes_slices)]
    return result


def _apply_dynamic_slice(
    input_tensor: torch.Tensor,
    starts: Tuple[ShapeDimension, ...],
    ends: Tuple[ShapeDimension, ...],
    axes: Tuple[int, ...],
    steps: Tuple[ShapeDimension, ...],
    constant_steps: Optional[Tuple[int, ...]],
) -> torch.Tensor:
    result = input_tensor
    rank = result.dim()

    for index, (start, end, axis, step_value) in enumerate(
        zip(starts, ends, axes, steps)
    ):
        dim = axis if axis >= 0 else axis + rank
        if dim < 0 or dim >= rank:
            raise IndexError(f"Slice axis {axis} is out of range for rank {rank}.")

        if _is_symbolic(step_value):
            if constant_steps is not None and index < len(constant_steps):
                step_value = constant_steps[index]
            else:
                raise NotImplementedError(
                    "Dynamic Slice requires concrete step values; provide Constant 'steps'."
                )

        step_int = int(step_value)
        if step_int == 0:
            warnings.warn(
                "Slice step of 0 encountered; treating as 1 for compatibility.",
                RuntimeWarning,
                stacklevel=2,
            )
            step_int = 1

        if step_int != 1:
            raise NotImplementedError("Dynamic Slice currently supports only step=1.")

        torch.ops.aten.sym_constrain_range.default(start, min=0)
        torch.ops.aten.sym_constrain_range.default(end - start, min=0)
        torch._check(start >= 0)
        torch._check(end >= start)

        result = torch.ops.aten.slice.Tensor(result, dim, start, end, step_int)

    return result


def _maybe_constant_sequence(values: Optional[np.ndarray]) -> Optional[Tuple[int, ...]]:
    if values is None:
        return None
    flattened = values.reshape(-1).tolist()
    return tuple(int(value) for value in flattened)


class OnnxSliceV9(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    def __init__(
        self, starts: np.ndarray, ends: np.ndarray, axes: Optional[np.ndarray] = None
    ):
        super().__init__()

        starts_tuple = tuple(int(value) for value in starts.reshape(-1).tolist())
        ends_tuple = tuple(int(value) for value in ends.reshape(-1).tolist())

        if axes is None:
            axes_tuple = tuple(range(len(starts_tuple)))
        else:
            axes_tuple = tuple(int(value) for value in axes.reshape(-1).tolist())

        steps_tuple = tuple(1 for _ in starts_tuple)
        (
            self._flip_dims,
            self._pos_axes_slices,
            self._neg_axes_slices,
        ) = _materialise_static_slice(starts_tuple, ends_tuple, axes_tuple, steps_tuple)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        return _execute_static_slice(
            input_tensor,
            self._flip_dims,
            self._pos_axes_slices,
            self._neg_axes_slices,
        )


class OnnxSlice(nn.Module, OnnxToTorchModuleWithCustomExport):  # pylint: disable=missing-class-docstring
    def __init__(
        self,
        constant_axes: Optional[np.ndarray] = None,
        constant_steps: Optional[np.ndarray] = None,
        constant_starts: Optional[np.ndarray] = None,
        constant_ends: Optional[np.ndarray] = None,
    ):
        super().__init__()

        self._constant_axes = _maybe_constant_sequence(constant_axes)
        self._constant_steps = _maybe_constant_sequence(constant_steps)
        self._constant_starts = _maybe_constant_sequence(constant_starts)
        self._constant_ends = _maybe_constant_sequence(constant_ends)
        self._cached_static_slice: Optional[
            Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]
        ] = None

    def _resolve_axes(
        self,
        axes: Optional[torch.Tensor],
        *,
        length: int,
    ) -> Tuple[int, ...]:
        if axes is not None:
            axes_sequence = sequence_to_symint_tuple(axes)
            if _all_concrete(axes_sequence):
                return _coerce_int_tuple(axes_sequence, description="axes")

            if self._constant_axes is not None:
                if len(self._constant_axes) != length:
                    raise ValueError(
                        "Slice constant axes length does not match starts length."
                    )
                return self._constant_axes

            raise NotImplementedError(
                "Slice axes requires concrete integers; received a symbolic value."
            )

        if self._constant_axes is not None:
            return self._constant_axes

        return tuple(range(length))

    def _resolve_steps(
        self,
        steps: Optional[torch.Tensor],
        *,
        length: int,
    ) -> Tuple[ShapeDimension, ...]:
        if steps is None:
            if self._constant_steps is not None:
                return self._constant_steps
            return tuple(1 for _ in range(length))

        resolved = sequence_to_symint_tuple(steps)
        if len(resolved) != length:
            raise ValueError(
                f"Slice steps length {len(resolved)} does not match starts length {length}."
            )
        return resolved

    def forward(  # pylint: disable=missing-function-docstring, too-many-branches
        self,
        input_tensor: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
        axes: Optional[torch.Tensor] = None,
        steps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        def _forward() -> torch.Tensor:
            starts_tuple = sequence_to_symint_tuple(starts)
            ends_tuple = sequence_to_symint_tuple(ends)
            if self._constant_starts is not None and not _all_concrete(starts_tuple):
                starts_tuple = self._constant_starts
            if self._constant_ends is not None and not _all_concrete(ends_tuple):
                ends_tuple = self._constant_ends

            if len(starts_tuple) != len(ends_tuple):
                raise ValueError("Slice starts and ends must have the same length.")

            axes_tuple = self._resolve_axes(axes, length=len(starts_tuple))

            if len(axes_tuple) != len(starts_tuple):
                raise ValueError("Slice axes length must match starts length.")

            steps_tuple = self._resolve_steps(steps, length=len(starts_tuple))

            if (
                _all_concrete(starts_tuple)
                and _all_concrete(ends_tuple)
                and _all_concrete(steps_tuple)
            ):
                starts_int = _coerce_int_tuple(starts_tuple, description="starts")
                ends_int = _coerce_int_tuple(ends_tuple, description="ends")
                steps_int = _coerce_int_tuple(steps_tuple, description="steps")

                (
                    flip_dims,
                    pos_axes_slices,
                    neg_axes_slices,
                ) = _materialise_static_slice(
                    starts_int, ends_int, axes_tuple, steps_int
                )
                self._cached_static_slice = (
                    starts_int,
                    ends_int,
                    axes_tuple,
                    steps_int,
                )
                return _execute_static_slice(
                    input_tensor, flip_dims, pos_axes_slices, neg_axes_slices
                )

            if self._cached_static_slice is not None:
                (
                    cached_starts,
                    cached_ends,
                    cached_axes,
                    cached_steps,
                ) = self._cached_static_slice
                (
                    flip_dims,
                    pos_axes_slices,
                    neg_axes_slices,
                ) = _materialise_static_slice(
                    cached_starts, cached_ends, cached_axes, cached_steps
                )
                return _execute_static_slice(
                    input_tensor, flip_dims, pos_axes_slices, neg_axes_slices
                )

            return _apply_dynamic_slice(
                input_tensor,
                starts_tuple,
                ends_tuple,
                axes_tuple,
                steps_tuple,
                constant_steps=self._constant_steps,
            )

        if torch.onnx.is_in_onnx_export():
            args = [input_tensor, starts, ends]
            if axes is not None:
                args.append(axes)
            if steps is not None:
                args.append(steps)

            return DefaultExportToOnnx.export(_forward, "Slice", *args, {})

        return _forward()


@add_converter(operation_type="Slice", version=9)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    node_attributes = node.attributes
    return OperationConverterResult(
        torch_module=OnnxSliceV9(
            starts=node_attributes["starts"],
            ends=node_attributes["ends"],
            axes=node_attributes.get("axes", None),
        ),
        onnx_mapping=onnx_mapping_from_node(node),
    )


@add_converter(operation_type="Slice", version=10)
@add_converter(operation_type="Slice", version=11)
@add_converter(operation_type="Slice", version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    constant_axes = None
    constant_steps = None
    constant_starts = None
    constant_ends = None

    def _maybe_constant(value_name: str) -> Optional[np.ndarray]:
        value_type = graph.value_type(value_name)

        if value_type == ValueType.GRAPH_INITIALIZER:
            return graph.initializers[value_name].to_numpy()

        if value_type == ValueType.NODE_OUTPUT:
            producer, _ = graph.value_as_node_output(value_name)
            if producer.operation_type == "Constant":
                constant_value = producer.attributes.get("value")
                if isinstance(constant_value, OnnxTensor):
                    return constant_value.to_numpy()
                if isinstance(constant_value, list):
                    return np.array(constant_value)
                if constant_value is not None:
                    return np.array([constant_value])

        return None

    if len(node.input_values) >= 2:
        constant_starts = _maybe_constant(node.input_values[1])

    if len(node.input_values) >= 3:
        constant_ends = _maybe_constant(node.input_values[2])

    if len(node.input_values) >= 4:
        axes_name = node.input_values[3]
        constant_axes = _maybe_constant(axes_name)

    if len(node.input_values) >= 5:
        steps_name = node.input_values[4]
        constant_steps = _maybe_constant(steps_name)

    return OperationConverterResult(
        torch_module=OnnxSlice(
            constant_axes=constant_axes,
            constant_steps=constant_steps,
            constant_starts=constant_starts,
            constant_ends=constant_ends,
        ),
        onnx_mapping=onnx_mapping_from_node(node),
    )
