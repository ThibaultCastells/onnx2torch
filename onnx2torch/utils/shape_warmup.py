"""Utilities to execute models with cheap placeholders to materialise shapes."""

from __future__ import annotations

import contextvars
import logging
import math
from contextlib import contextmanager
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Tuple

import torch
from torch.utils._python_dispatch import TorchDispatchMode


LOGGER = logging.getLogger("onnx2torch.shape_warmup")


def _broadcast_shape(*shapes: Iterable[int]) -> Tuple[int, ...]:
    expanded = [tuple(int(dim) for dim in shape) for shape in shapes]
    try:
        return torch.broadcast_shapes(*expanded)
    except RuntimeError:
        max_rank = max((len(shape) for shape in expanded), default=0)
        result = []
        for axis in range(1, max_rank + 1):
            dims = []
            for shape in expanded:
                if len(shape) >= axis:
                    dims.append(int(shape[-axis]))
                else:
                    dims.append(1)

            resolved = 1
            for dim in dims:
                candidate = max(1, abs(dim))
                if candidate > resolved:
                    resolved = candidate
            result.append(resolved)

        return tuple(reversed(result))


def _zero_tensor(
    shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return torch.zeros(shape, dtype=dtype, device=device)


def _matmul_shape(lhs: torch.Tensor, rhs: torch.Tensor) -> Tuple[int, ...]:
    if lhs.dim() == 1 and rhs.dim() == 1:
        return ()
    if lhs.dim() == 2 and rhs.dim() == 2:
        return (lhs.size(0), rhs.size(1))

    # Batched matmul path. Follow PyTorch broadcasting semantics.
    lhs_batch = lhs.shape[:-2]
    rhs_batch = rhs.shape[:-2]
    batch_shape = _broadcast_shape(lhs_batch, rhs_batch)
    return batch_shape + (lhs.size(-2), rhs.size(-1))


def _return_zero_like(reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(reference)


def _result_type(*tensors: torch.Tensor) -> torch.dtype:
    dtypes = [tensor.dtype for tensor in tensors if isinstance(tensor, torch.Tensor)]
    if not dtypes:
        return torch.float32

    result = dtypes[0]
    for dtype in dtypes[1:]:
        result = torch.promote_types(result, dtype)
    return result


def _result_device(*tensors: torch.Tensor) -> torch.device:
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            return tensor.device
    return torch.device("cpu")


class ShapeWarmupMode(TorchDispatchMode):
    """Dispatch mode that short-circuits heavy ops with zero placeholders."""

    _UNARY_ZERO = {
        torch.ops.aten.relu.default,
        torch.ops.aten.gelu.default,
        torch.ops.aten.tanh.default,
        torch.ops.aten.sigmoid.default,
        torch.ops.aten.silu.default,
        torch.ops.aten.log.default,
        torch.ops.aten.exp.default,
        torch.ops.aten.abs.default,
        torch.ops.aten.neg.default,
        torch.ops.aten.sqrt.default,
        torch.ops.aten.rsqrt.default,
        torch.ops.aten.sinh.default,
        torch.ops.aten.cosh.default,
        torch.ops.aten.softplus.default,
        torch.ops.aten.cos.default,
        torch.ops.aten.sin.default,
    }

    _BINARY_ZERO = {
        torch.ops.aten.add.Tensor,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.div.Tensor,
        torch.ops.aten.maximum.default,
        torch.ops.aten.minimum.default,
        torch.ops.aten.pow.Tensor_Tensor,
    }

    _COMPARISON_ZERO = {
        torch.ops.aten.less_equal.Tensor,
        torch.ops.aten.less.Tensor,
        torch.ops.aten.greater.Tensor,
        torch.ops.aten.greater_equal.Tensor,
    }

    _LOGICAL_ZERO = {
        torch.ops.aten.logical_and.default,
        torch.ops.aten.logical_or.default,
        torch.ops.aten.logical_xor.default,
    }

    _MATMUL_OPS = {
        torch.ops.aten.matmul.default,
        torch.ops.aten.mm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.addmm.out,
    }

    _SOFTMAX_OPS = {
        torch.ops.aten.softmax.int,
        torch.ops.aten.log_softmax.int,
    }

    _DROPOUT_OPS = {
        torch.ops.aten.dropout.default,
        torch.ops.aten.native_dropout.default,
    }

    _INPLACE_DROPOUT = {
        torch.ops.aten.dropout_.default,
    }

    _NATIVE_LAYER_NORM = {
        torch.ops.aten.native_layer_norm.default,
    }

    _LAYER_NORM = {
        torch.ops.aten.layer_norm.default,
    }

    _LINEAR_OPS = {
        torch.ops.aten.linear.default,
    }

    _INDEX_SELECT_OPS = {
        torch.ops.aten.index_select.default,
    }

    _RESHAPE_OPS = {
        torch.ops.aten.view.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._reshape_alias.default,
    }

    _EXPAND_OPS = {
        torch.ops.aten.expand.default,
    }

    _WHERE_OPS = {
        torch.ops.aten.where.self,
    }

    _CAT_OPS = {
        torch.ops.aten.cat.default,
    }

    _MEAN_OPS = {
        torch.ops.aten.mean.dim,
    }

    def __torch_dispatch__(
        self,
        func: torch._ops.OpOverload,
        types: Tuple[type, ...],
        args: Tuple[Any, ...] = (),
        kwargs: Dict[str, Any] | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        if func in self._UNARY_ZERO:
            reference = args[0]
            return _return_zero_like(reference)

        if func in self._BINARY_ZERO:
            lhs, rhs = args[0], args[1]
            shape = _broadcast_shape(lhs.shape, rhs.shape)
            dtype = _result_type(lhs, rhs)
            device = _result_device(lhs, rhs)
            return _zero_tensor(shape, dtype=dtype, device=device)

        if func in self._COMPARISON_ZERO:
            lhs, rhs = args[0], args[1]
            shape = _broadcast_shape(lhs.shape, rhs.shape)
            device = _result_device(lhs, rhs)
            return _zero_tensor(shape, dtype=torch.bool, device=device)

        if func in self._LOGICAL_ZERO:
            lhs, rhs = args[0], args[1]
            shape = _broadcast_shape(lhs.shape, rhs.shape)
            device = _result_device(lhs, rhs)
            return _zero_tensor(shape, dtype=torch.bool, device=device)

        if func in self._MATMUL_OPS:
            lhs, rhs = args[0], args[1]
            shape = _matmul_shape(lhs, rhs)
            dtype = _result_type(lhs, rhs)
            device = _result_device(lhs, rhs)
            return _zero_tensor(shape, dtype=dtype, device=device)

        if func in self._LINEAR_OPS:
            input_tensor, weight = args[0], args[1]
            bias = args[2] if len(args) > 2 else kwargs.get("bias")
            dtype = _result_type(input_tensor, weight)
            if bias is not None:
                dtype = _result_type(
                    _zero_tensor((), dtype=dtype, device=input_tensor.device), bias
                )
            output_shape = input_tensor.shape[:-1] + (weight.shape[0],)
            return _zero_tensor(output_shape, dtype=dtype, device=input_tensor.device)

        if func in self._SOFTMAX_OPS:
            reference = args[0]
            return _return_zero_like(reference)

        if func in self._DROPOUT_OPS:
            reference = args[0]
            mask = torch.zeros_like(reference, dtype=torch.bool)
            return _return_zero_like(reference), mask

        if func in self._INPLACE_DROPOUT:
            # In-place dropout returns the input tensor.
            return args[0]

        if func in self._NATIVE_LAYER_NORM:
            input_tensor = args[0]
            normalized_shape = input_tensor.shape
            normalized = _zero_tensor(
                normalized_shape, dtype=input_tensor.dtype, device=input_tensor.device
            )
            mean = torch.zeros(
                normalized_shape[:-1],
                dtype=input_tensor.dtype,
                device=input_tensor.device,
            )
            rstd = torch.ones_like(mean)
            return normalized, mean, rstd

        if func in self._LAYER_NORM:
            input_tensor = args[0]
            return _zero_tensor(
                input_tensor.shape, dtype=input_tensor.dtype, device=input_tensor.device
            )

        if func in self._RESHAPE_OPS:
            input_tensor = args[0]
            if func is torch.ops.aten._reshape_alias.default:
                shape = args[1]
            else:
                shape = args[1]

            if isinstance(shape, torch.Tensor):
                shape_sequence = [int(dim) for dim in shape.tolist()]
            else:
                shape_sequence = []
                for dim in shape:
                    if isinstance(dim, torch.Tensor):
                        shape_sequence.append(int(dim.item()))
                    else:
                        shape_sequence.append(int(dim))

            if -1 in shape_sequence:
                unknown_index = shape_sequence.index(-1)
                known_product = math.prod(
                    dim if idx != unknown_index else 1
                    for idx, dim in enumerate(shape_sequence)
                )
                total_elements = int(input_tensor.numel())
                fill_value = 0
                if known_product != 0:
                    fill_value = total_elements // known_product
                shape_sequence[unknown_index] = fill_value

            output_shape = tuple(shape_sequence)
            return _zero_tensor(
                output_shape, dtype=input_tensor.dtype, device=input_tensor.device
            )

        if func in self._INDEX_SELECT_OPS:
            input_tensor, dim, index = args[:3]

            if isinstance(dim, torch.Tensor):
                dim = int(dim.item())
            else:
                dim = int(dim)

            dim = dim % input_tensor.dim()
            index_shape = tuple(int(v) for v in index.shape)
            output_shape = (
                input_tensor.shape[:dim] + index_shape + input_tensor.shape[dim + 1 :]
            )

            return _zero_tensor(
                output_shape, dtype=input_tensor.dtype, device=input_tensor.device
            )

        if func in self._EXPAND_OPS:
            input_tensor = args[0]
            target_shape = args[1]

            if isinstance(target_shape, torch.Tensor):
                raw_shape = [int(dim) for dim in target_shape.tolist()]
            else:
                raw_shape = []
                for dim in target_shape:
                    if isinstance(dim, torch.Tensor):
                        raw_shape.append(int(dim.item()))
                    else:
                        raw_shape.append(int(dim))

            input_shape = tuple(int(dim) for dim in input_tensor.shape)
            output_shape = list(raw_shape)
            offset = len(output_shape) - len(input_shape)
            for idx, dim in enumerate(output_shape):
                if dim == -1:
                    aligned_idx = idx - offset
                    if aligned_idx >= 0:
                        replacement = input_shape[aligned_idx]
                    else:
                        replacement = 1
                    output_shape[idx] = replacement

            for idx, dim in enumerate(output_shape):
                if dim <= 0:
                    aligned_idx = idx - offset
                    fallback = 1
                    if aligned_idx >= 0 and aligned_idx < len(input_shape):
                        fallback = max(1, input_shape[aligned_idx])
                    output_shape[idx] = fallback

            return _zero_tensor(
                tuple(output_shape),
                dtype=input_tensor.dtype,
                device=input_tensor.device,
            )

        if func in self._WHERE_OPS:
            condition, input_tensor, other = args[:3]

            def _shape(value: Any) -> Tuple[int, ...]:
                if isinstance(value, torch.Tensor):
                    return tuple(int(dim) for dim in value.shape)
                return ()

            shape = _broadcast_shape(
                _shape(condition), _shape(input_tensor), _shape(other)
            )

            tensors_for_type = [
                value
                for value in (input_tensor, other)
                if isinstance(value, torch.Tensor)
            ]
            if tensors_for_type:
                dtype = _result_type(*tensors_for_type)
                device = _result_device(*tensors_for_type)
            else:
                dtype = torch.float32
                device = torch.device("cpu")
            return _zero_tensor(shape, dtype=dtype, device=device)

        if func in self._CAT_OPS:
            tensors = tuple(args[0])
            if not tensors:
                return torch.zeros(0, device="cpu")

            dim = 0
            if len(args) > 1:
                dim = int(args[1])
            elif "dim" in kwargs:
                dim = int(kwargs["dim"])

            reference = tensors[0]
            rank = reference.dim()
            if dim < 0:
                dim += rank
            dim = max(0, min(rank - 1, dim))

            shapes = [list(t.shape) for t in tensors]
            base_shape = list(shapes[0])
            total = 0
            for shape in shapes:
                size = 1
                if dim < len(shape):
                    size = max(1, int(shape[dim]))
                total += size
            base_shape[dim] = total

            dtype = _result_type(*tensors)
            device = _result_device(*tensors)
            return _zero_tensor(tuple(base_shape), dtype=dtype, device=device)

        if func in self._MEAN_OPS:
            input_tensor = args[0]
            dims_arg = args[1] if len(args) > 1 else kwargs.get("dim", ())
            keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
            dtype_override = None
            if len(args) > 3:
                dtype_override = args[3]
            elif "dtype" in kwargs:
                dtype_override = kwargs["dtype"]

            if isinstance(dims_arg, int):
                dims = [dims_arg]
            else:
                dims = list(dims_arg)

            rank = input_tensor.dim()
            normalized_dims = sorted({int(dim) % rank for dim in dims})
            output_shape = list(input_tensor.shape)
            if keepdim:
                for dim in normalized_dims:
                    output_shape[dim] = 1
            else:
                for offset, dim in enumerate(normalized_dims):
                    output_shape.pop(dim - offset)

            dtype = dtype_override or input_tensor.dtype
            return _zero_tensor(
                tuple(output_shape), dtype=dtype, device=input_tensor.device
            )

        LOGGER.debug("Delegating to real op during shape warmup: %s", func)
        return func(*args, **kwargs)


_SHAPE_WARMUP_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "shape_warmup_active", default=False
)


def is_shape_warmup_active() -> bool:
    """Return True when the shape warmup dispatch mode is active."""
    if getattr(torch, "_dynamo", None) is not None and torch._dynamo.is_compiling():
        return False

    return _SHAPE_WARMUP_ACTIVE.get()


@contextmanager
def shape_warmup():
    token = _SHAPE_WARMUP_ACTIVE.set(True)
    try:
        mode = ShapeWarmupMode()
        with mode:
            yield
    finally:
        _SHAPE_WARMUP_ACTIVE.reset(token)
