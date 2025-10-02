"""Helpers for working with ONNX shape tensors and symbolic dimensions."""

from __future__ import annotations

from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch
from torch.utils._python_dispatch import _disable_current_modes

try:  # pragma: no cover - FakeTensor may not exist on older torch
    from torch._subclasses.fake_tensor import (
        DataDependentOutputException,
        FakeTensor,
    )
except ImportError:  # pragma: no cover - keep isinstance checks safe
    FakeTensor = ()  # type: ignore[assignment]
    DataDependentOutputException = RuntimeError  # type: ignore[assignment]

try:  # pragma: no cover - SymInt introduced in newer torch releases
    from torch import SymInt  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fall back when SymInt is missing
    SymInt = int  # type: ignore[assignment]

from onnx2torch.utils.shape_warmup import is_shape_warmup_active

ShapeDimension = Union[int, SymInt]


def _flatten_tensor(values: torch.Tensor) -> torch.Tensor:
    target_shape: Tuple[int, ...]
    if values.ndim == 0:
        target_shape = (1,)
    else:
        target_shape = (-1,)

    if is_shape_warmup_active() and not isinstance(values, FakeTensor):
        with _disable_current_modes():
            return values.reshape(target_shape)

    return values.reshape(target_shape)


def _tensor_to_symint_tuple(tensor: torch.Tensor) -> Tuple[ShapeDimension, ...]:
    if is_shape_warmup_active() and not isinstance(tensor, FakeTensor):
        with _disable_current_modes():
            canonical = _flatten_tensor(tensor.to(dtype=torch.int64))
    else:
        canonical = _flatten_tensor(tensor.to(dtype=torch.int64))

    if isinstance(canonical, FakeTensor):
        flattened = canonical.reshape(-1)
        result: List[ShapeDimension] = []
        for element in flattened:
            if not isinstance(element, FakeTensor):
                result.append(int(element.item()))
                continue

            try:
                value = element.item()
            except (
                DataDependentOutputException
            ) as error:  # pragma: no cover - missing ShapeEnv
                raise NotImplementedError(
                    "Failed to materialise symbolic shape values; ensure a ShapeEnv is available."
                ) from error

            result.append(value)

        return tuple(result)

    try:
        if is_shape_warmup_active() and not isinstance(canonical, FakeTensor):
            with _disable_current_modes():
                values = canonical.tolist()
        else:
            values = canonical.tolist()
    except Exception as error:  # pragma: no cover - provide actionable failure
        raise NotImplementedError(
            "Failed to materialise symbolic shape values; ensure a ShapeEnv is available."
        ) from error

    return tuple(values)


def shape_tensor_to_sequence(shape: torch.Tensor) -> Tuple[ShapeDimension, ...]:
    """Return a tuple of shape dimensions safe for eager and fake tensor modes."""

    if shape.numel() == 0:
        return ()

    return _tensor_to_symint_tuple(shape)


SequenceInput = Union[torch.Tensor, np.ndarray, Sequence[Union[int, SymInt]]]


def sequence_to_symint_tuple(values: SequenceInput) -> Tuple[ShapeDimension, ...]:
    """Coerce ONNX-provided sequences to a tuple of SymInt-friendly scalars."""

    if isinstance(values, torch.Tensor):
        return _tensor_to_symint_tuple(values)

    if isinstance(values, np.ndarray):
        return tuple(int(value) for value in values.reshape(-1).tolist())

    return tuple(values)
