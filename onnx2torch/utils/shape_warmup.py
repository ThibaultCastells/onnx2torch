"""Utilities to execute models with cheap placeholders to materialise shapes."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Tuple

import torch


def _broadcast_shape(*shapes: Iterable[int]) -> Tuple[int, ...]:
    expanded = [tuple(int(dim) for dim in shape) for shape in shapes]
    return torch.broadcast_shapes(*expanded)


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
    return torch.result_type(*tensors)


def _result_device(*tensors: torch.Tensor) -> torch.device:
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            return tensor.device
    return torch.device("cpu")


class ShapeWarmupMode(torch.utils._python_dispatch.TorchDispatchMode):
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

        return func(*args, **kwargs)


@contextmanager
def shape_warmup():
    mode = ShapeWarmupMode()
    with mode:
        yield
