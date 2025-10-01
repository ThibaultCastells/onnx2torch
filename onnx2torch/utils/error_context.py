"""Helpers for attaching ONNX debugging context to converter modules."""

from __future__ import annotations

from functools import wraps
from types import MethodType

from torch import nn

from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxMapping
from onnx2torch.utils.common import OnnxToTorchModule

_ContextNote = str


def attach_onnx_context(
    module: nn.Module,
    node: OnnxNode,
    mapping: OnnxMapping,
) -> None:
    """Enrich ``module`` with ONNX metadata and error notes."""
    if not isinstance(module, OnnxToTorchModule):
        return

    if getattr(module, "_onnx_context_attached", False):
        return

    original_forward = module.forward

    @wraps(original_forward)
    def forward_with_context(self, *args, **kwargs):  # type: ignore[override]
        try:
            return original_forward(*args, **kwargs)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _add_onnx_note(exc, node, mapping)
            raise

    module.forward = MethodType(forward_with_context, module)
    setattr(module, "_onnx_context_attached", True)


def _add_onnx_note(exc: Exception, node: OnnxNode, mapping: OnnxMapping) -> None:
    note = _build_note(node, mapping)
    _attach_note(exc, note)


def _build_note(node: OnnxNode, mapping: OnnxMapping) -> _ContextNote:
    name = node.name or "<unnamed>"
    parts = [
        "ONNX context:",
        f"  op_type: {node.operation_type}",
        f"  node_name: {name}",
        f"  node_id: {node.unique_name}",
    ]

    if node.domain:
        parts.append(f"  domain: {node.domain or 'ai.onnx'}")

    if mapping.inputs:
        joined_inputs = ", ".join(mapping.inputs)
        parts.append(f"  inputs: {joined_inputs}")

    if mapping.outputs:
        joined_outputs = ", ".join(mapping.outputs)
        parts.append(f"  outputs: {joined_outputs}")

    return "\n".join(parts)


def _attach_note(exc: Exception, note: _ContextNote) -> None:
    if hasattr(exc, "add_note"):
        exc.add_note(note)
        return

    _append_to_args(exc, note)


def _append_to_args(exc: Exception, note: _ContextNote) -> None:
    message, *rest = getattr(exc, "args", (note,))
    exc.args = tuple([message] + rest + [note])
