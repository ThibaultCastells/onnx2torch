"""Helpers for working with ONNX shape tensors."""

from __future__ import annotations

from typing import Tuple
from typing import Union

import torch

try:  # pragma: no cover - optional dependency in older torch releases
    from torch._dynamo import is_compiling as _dynamo_is_compiling
except ImportError:  # pragma: no cover - dynamo unavailable

    def _dynamo_is_compiling() -> bool:
        return False


try:  # pragma: no cover - FakeTensor may not exist on older torch
    from torch._subclasses.fake_tensor import FakeTensor
except ImportError:  # pragma: no cover - keep isinstance checks safe
    FakeTensor = ()  # type: ignore[assignment]

try:  # pragma: no cover - SymInt introduced in newer torch releases
    from torch import SymInt  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fall back when SymInt is missing
    SymInt = int  # type: ignore[assignment]

ShapeDimension = Union[int, SymInt, torch.Tensor]


def shape_tensor_to_sequence(shape: torch.Tensor) -> Tuple[ShapeDimension, ...]:
    """Return a tuple of shape dimensions safe for eager and fake tensor modes."""

    shape_tensor = shape.to(dtype=torch.int64)

    if shape_tensor.numel() == 0:
        return ()

    flattened = shape_tensor.reshape(-1)

    if isinstance(shape, FakeTensor):
        try:
            values = flattened.tolist()
        except (
            Exception
        ) as error:  # pragma: no cover - defensive against missing shape env
            raise NotImplementedError(
                "FakeTensor shapes without a shape environment are not supported"
            ) from error
        return tuple(values)

    if _dynamo_is_compiling():
        return tuple(flattened.unbind(0))

    return tuple(int(dimension) for dimension in flattened.tolist())
