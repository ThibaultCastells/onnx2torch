"""Helpers for working with ONNX shape tensors and symbolic dimensions."""

from __future__ import annotations

from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch

try:  # pragma: no cover - FakeTensor may not exist on older torch
    from torch._subclasses.fake_tensor import FakeTensor
except ImportError:  # pragma: no cover - keep isinstance checks safe
    FakeTensor = ()  # type: ignore[assignment]

try:  # pragma: no cover - SymInt introduced in newer torch releases
    from torch import SymInt  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fall back when SymInt is missing
    SymInt = int  # type: ignore[assignment]

ShapeDimension = Union[int, SymInt]


def _flatten_tensor(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 0:
        return values.reshape(1)
    return values.reshape(-1)


def _tensor_to_symint_tuple(tensor: torch.Tensor) -> Tuple[ShapeDimension, ...]:
    canonical = _flatten_tensor(tensor.to(dtype=torch.int64))

    if isinstance(canonical, FakeTensor):
        fake_mode = getattr(canonical, "fake_mode", None)
        shape_env = getattr(fake_mode, "shape_env", None)
        if shape_env is None:  # pragma: no cover - symbolic path requires ShapeEnv
            raise NotImplementedError(
                "Failed to materialise symbolic shape values; ensure a ShapeEnv is available."
            )

        flattened = canonical.reshape(-1)
        result: List[ShapeDimension] = []
        for element in flattened:
            if isinstance(element, FakeTensor):
                # FakeTensor scalars carry SymInt metadata via the current ShapeEnv.
                symint = shape_env.create_unbacked_symint()
                try:  # preserve available runtime hint for better guards
                    hint = int(element.item())
                except Exception:  # pragma: no cover - FakeTensor without item support
                    hint = None
                if hint is not None and hasattr(symint, "node"):
                    symint.node._hint = hint  # type: ignore[attr-defined]
                result.append(symint)
            else:
                result.append(int(element.item()))
        return tuple(result)

    try:
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
