from __future__ import annotations

# pylint: disable=missing-docstring
__all__ = [
    "OnnxIf",
]

from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import torch
from onnx import defs
from onnx import helper
from onnx.onnx_ml_pb2 import GraphProto
from onnx.onnx_ml_pb2 import ModelProto
from onnx.onnx_ml_pb2 import ValueInfoProto
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_graph import ValueType
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxMapping
from onnx2torch.utils.common import get_const_value
from onnx2torch.utils.common import get_onnx_version
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.custom_export_to_onnx import CustomExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport

try:  # pragma: no cover - torch internals may move
    from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode
except ImportError:  # pragma: no cover - fallback for older torch
    GuardOnDataDependentSymNode = RuntimeError  # type: ignore[assignment]

try:  # pragma: no cover - FakeTensor may not exist
    from torch._subclasses.fake_tensor import DataDependentOutputException  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fallback
    DataDependentOutputException = RuntimeError  # type: ignore[assignment]

try:  # pragma: no cover - torch internals may move
    from torch._dynamo.exc import UncapturedHigherOrderOpError  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fallback for older torch
    class UncapturedHigherOrderOpError(RuntimeError):  # type: ignore[too-many-ancestors]
        """Fallback placeholder when torch._dynamo.exc is unavailable."""


def _copy_graph(graph_proto: GraphProto) -> GraphProto:
    copied = GraphProto()
    copied.CopyFrom(graph_proto)
    return copied


def _collect_external_inputs(graph_proto: GraphProto) -> List[str]:
    inputs = [value_info.name for value_info in graph_proto.input if value_info.name]
    initializers = {initializer.name for initializer in graph_proto.initializer}

    seen: set[str] = set(inputs)
    seen.update(initializers)

    external: List[str] = []

    for node in graph_proto.node:
        for name in node.input:
            if not name or name in seen:
                continue
            if name not in external:
                external.append(name)
        for name in node.output:
            if name:
                seen.add(name)

    return external


def _augment_branch_inputs(
    graph_proto: GraphProto,
    new_inputs: Iterable[str],
) -> List[str]:
    input_names = [
        value_info.name for value_info in graph_proto.input if value_info.name
    ]
    existing = set(input_names)

    for name in new_inputs:
        if not name or name in existing:
            continue
        value_info = ValueInfoProto()
        value_info.name = name
        graph_proto.input.append(value_info)
        input_names.append(name)
        existing.add(name)

    return input_names


def _make_branch_model(
    graph_proto: GraphProto,
    ir_version: int | None,
    opset_import: Dict[str, int],
) -> ModelProto:
    model = helper.make_model(graph_proto, producer_name="onnx2torch.if.branch")
    model.ir_version = ir_version or defs.IR_VERSION
    model.opset_import.clear()
    for domain, version in opset_import.items():
        op_import = model.opset_import.add()
        op_import.domain = domain
        op_import.version = version
    if not model.opset_import:
        op_import = model.opset_import.add()
        op_import.domain = defs.ONNX_DOMAIN
        op_import.version = defs.onnx_opset_version()
    return model


def _convert_branch(
    graph_proto: GraphProto,
    onnx_graph: OnnxGraph,
) -> Tuple[nn.Module, List[str]]:
    branch_graph = _copy_graph(graph_proto)
    external_inputs = _collect_external_inputs(branch_graph)

    runtime_inputs: List[str] = []
    branch_initializer_names = {initializer.name for initializer in branch_graph.initializer}

    for name in external_inputs:
        if not name:
            continue
        if name in onnx_graph.initializers:
            if name in branch_initializer_names:
                continue
            initializer_proto = branch_graph.initializer.add()
            initializer_proto.CopyFrom(onnx_graph.initializers[name].proto)
            branch_initializer_names.add(name)
        else:
            runtime_inputs.append(name)

    branch_inputs = _augment_branch_inputs(branch_graph, runtime_inputs)
    branch_inputs = [name for name in branch_inputs if name]

    opset_import = getattr(onnx_graph, "opset_import", {})
    ir_version = getattr(onnx_graph, "ir_version", None)
    branch_model = _make_branch_model(branch_graph, ir_version, opset_import)

    # Import locally to avoid circular dependency during module import time.
    from onnx2torch.converter import convert as convert_model  # pylint: disable=cyclic-import

    torch_module = convert_model(branch_model)
    torch_module = torch_module.eval()
    return torch_module, branch_inputs


class IfExportToOnnx(CustomExportToOnnx):
    @staticmethod
    def symbolic(
        graph,  # type: ignore[override]
        op_type: str,
        condition,
        onnx_attrs: Dict[str, GraphProto],
        num_outputs: int,
    ):
        if num_outputs == 0:
            graph.op(op_type, condition, **onnx_attrs, outputs=0)
            return ()

        outputs = graph.op(op_type, condition, **onnx_attrs, outputs=num_outputs)
        return outputs


class OnnxIf(nn.Module, OnnxToTorchModuleWithCustomExport):
    def __init__(
        self,
        condition_input: str,
        captured_inputs: Sequence[str],
        then_module: nn.Module,
        else_module: nn.Module,
        then_inputs: Sequence[str],
        else_inputs: Sequence[str],
        then_branch_proto: GraphProto,
        else_branch_proto: GraphProto,
        num_outputs: int,
        condition_hint: Optional[bool] = None,
    ):
        super().__init__()
        self.condition_input = condition_input
        self._captured_inputs = list(captured_inputs)
        self._captured_index = {
            name: idx for idx, name in enumerate(self._captured_inputs)
        }
        self._num_outputs = num_outputs
        self._constant_condition = condition_hint

        self.then_branch_module = then_module
        self.else_branch_module = else_module

        self._then_arg_specs = self._build_arg_specs(then_inputs)
        self._else_arg_specs = self._build_arg_specs(else_inputs)

        self._then_branch_proto = then_branch_proto
        self._else_branch_proto = else_branch_proto

    def _build_arg_specs(
        self, input_names: Sequence[str]
    ) -> List[Tuple[bool, int | None]]:
        specs: List[Tuple[bool, int | None]] = []
        for name in input_names:
            if name == self.condition_input:
                specs.append((True, None))
            else:
                specs.append((False, self._captured_index.get(name)))
        return specs

    def _gather_args(
        self,
        specs: Sequence[Tuple[bool, int | None]],
        condition: torch.Tensor,
        captured: Sequence[torch.Tensor],
    ) -> List[torch.Tensor]:
        args: List[torch.Tensor] = []
        for is_condition, index in specs:
            if is_condition:
                args.append(condition)
            else:
                if index is None or index >= len(captured):
                    raise RuntimeError(
                        f'Missing captured input for branch (expected "{self._captured_inputs}")'
                    )
                args.append(captured[index])
        return args

    def _run_branch(
        self,
        module: nn.Module,
        specs: Sequence[Tuple[bool, int | None]],
        condition: torch.Tensor,
        captured: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        args = self._gather_args(specs, condition, captured)
        outputs = module(*args) if args else module()
        if isinstance(outputs, tuple):
            return outputs
        if outputs is None:
            return ()
        return (outputs,)

    @staticmethod
    def _normalize_condition(condition: torch.Tensor) -> torch.Tensor:
        if not isinstance(condition, torch.Tensor):
            condition = torch.as_tensor(condition)
        if condition.dtype != torch.bool:
            condition = condition.to(dtype=torch.bool)
        if condition.ndim != 0:
            condition = condition.reshape([])
        return condition

    def _evaluate_without_cond(
        self,
        condition: torch.Tensor,
        captured: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        condition_bool = self._normalize_condition(condition)
        use_then_branch = self._evaluate_condition_to_bool(condition_bool)
        module = self.then_branch_module if use_then_branch else self.else_branch_module
        specs = self._then_arg_specs if use_then_branch else self._else_arg_specs
        return self._run_branch(module, specs, condition_bool, captured)

    def _execute(
        self,
        condition: torch.Tensor,
        captured: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        condition_bool = self._normalize_condition(condition)

        if self._constant_condition is not None:
            return self._evaluate_with_hint(
                condition_bool,
                captured,
                self._constant_condition,
            )

        if hasattr(torch, "_dynamo") and torch._dynamo.is_compiling():
            hint = self._maybe_as_bool(condition_bool)
            if hint is not None:
                return self._evaluate_with_hint(condition_bool, captured, hint)
            return self._execute_with_fallbacks(condition_bool, captured)

        if hasattr(condition_bool, "node"):
            hint = self._maybe_as_bool(condition_bool)
            if hint is not None:
                return self._evaluate_with_hint(condition_bool, captured, hint)
            return self._execute_with_fallbacks(condition_bool, captured)

        try:
            return self._evaluate_without_cond(condition_bool, captured)
        except GuardOnDataDependentSymNode:
            return self._execute_with_fallbacks(condition_bool, captured)
        except RuntimeError as error:
            if "data-dependent expression" in str(error):
                return self._execute_with_fallbacks(condition_bool, captured)
            raise

    def _execute_with_fallbacks(
        self,
        condition_bool: torch.Tensor,
        captured: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        try:
            return self._execute_with_cond(condition_bool, captured)
        except GuardOnDataDependentSymNode:
            return self._evaluate_with_merge(condition_bool, captured)
        except RuntimeError as error:
            message = str(error)
            if "data-dependent expression" in message or "no rule registered" in message:
                return self._evaluate_with_merge(condition_bool, captured)
            raise

    @staticmethod
    def _maybe_as_bool(value: torch.Tensor) -> Optional[bool]:
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                try:
                    return bool(value.item())
                except DataDependentOutputException:
                    return None
                except GuardOnDataDependentSymNode:
                    return None
                except RuntimeError as error:
                    if "data-dependent expression" in str(error):
                        return None
                    raise

        maybe_bool: Optional[Callable[[], Optional[bool]]] = getattr(
            value, "maybe_as_bool", None
        )
        if callable(maybe_bool):
            hint = maybe_bool()
            if hint is not None:
                return bool(hint)

        node = getattr(value, "node", None)
        if node is not None:
            node_fn = getattr(node, "maybe_as_bool", None)
            if callable(node_fn):
                hint = node_fn()
                if hint is not None:
                    return bool(hint)

        return None

    def _evaluate_condition_to_bool(self, condition_bool: torch.Tensor) -> bool:
        try:
            return bool(condition_bool.item())
        except GuardOnDataDependentSymNode:
            hint = self._maybe_as_bool(condition_bool)
            if hint is not None:
                return hint
            raise
        except RuntimeError as error:
            if "data-dependent expression" not in str(error):
                raise
            hint = self._maybe_as_bool(condition_bool)
            if hint is not None:
                return hint
            raise

    def _execute_with_cond(
        self,
        condition_bool: torch.Tensor,
        captured: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        def then_fn() -> Tuple[torch.Tensor, ...]:
            outputs = self._run_branch(
                self.then_branch_module,
                self._then_arg_specs,
                condition_bool,
                captured,
            )
            return outputs

        def else_fn() -> Tuple[torch.Tensor, ...]:
            outputs = self._run_branch(
                self.else_branch_module,
                self._else_arg_specs,
                condition_bool,
                captured,
            )
            return outputs

        try:
            outputs = torch.cond(condition_bool, then_fn, else_fn)
        except GuardOnDataDependentSymNode:
            return self._evaluate_with_merge(condition_bool, captured)
        except UncapturedHigherOrderOpError:
            return self._evaluate_with_merge(condition_bool, captured)
        except NotImplementedError:
            return self._evaluate_with_merge(condition_bool, captured)
        except RuntimeError as error:
            if "data-dependent expression" in str(error):
                return self._evaluate_with_merge(condition_bool, captured)
            raise

        if isinstance(outputs, tuple):
            return outputs
        if outputs is None:
            return ()
        return (outputs,)

    def _evaluate_with_merge(
        self,
        condition_bool: torch.Tensor,
        captured: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        if self._num_outputs == 0:
            return ()

        if hasattr(torch, "_dynamo") and torch._dynamo.is_compiling():
            disable = getattr(torch._dynamo, "disable", None)
        else:
            disable = None

        def _evaluate_branches() -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
            then_eval = self._run_branch(
                self.then_branch_module,
                self._then_arg_specs,
                condition_bool,
                captured,
            )
            else_eval = self._run_branch(
                self.else_branch_module,
                self._else_arg_specs,
                condition_bool,
                captured,
            )
            return then_eval, else_eval

        ran_with_disable = False
        if disable is not None:
            if callable(disable):
                try:
                    decorated = disable(_evaluate_branches)
                    then_outputs, else_outputs = decorated()
                    ran_with_disable = True
                except TypeError:
                    pass

            if not ran_with_disable:
                disable_result = None
                try:
                    disable_result = disable()  # type: ignore[misc]
                except TypeError:
                    disable_result = None

                if (
                    disable_result is not None
                    and hasattr(disable_result, "__enter__")
                    and hasattr(disable_result, "__exit__")
                ):
                    with disable_result:
                        then_outputs, else_outputs = _evaluate_branches()
                    ran_with_disable = True

        if not ran_with_disable:
            then_outputs, else_outputs = _evaluate_branches()

        if len(then_outputs) != len(else_outputs):
            raise RuntimeError("If branches produced different number of outputs.")

        merged: List[torch.Tensor] = []
        for then_output, else_output in zip(then_outputs, else_outputs):
            if isinstance(then_output, torch.Tensor) and isinstance(
                else_output, torch.Tensor
            ):
                try:
                    merged.append(torch.where(condition_bool, then_output, else_output))
                except RuntimeError as error:
                    hint = self._maybe_as_bool(condition_bool)
                    if hint is not None:
                        merged.append(then_output if hint else else_output)
                        continue
                    raise error
                continue

            hint = self._maybe_as_bool(condition_bool)
            if hint is None:
                raise RuntimeError(
                    "Encountered non-tensor If outputs with symbolic condition; unable to merge."
                )
            merged.append(then_output if hint else else_output)

        if self._num_outputs == 1:
            return (merged[0],)
        return tuple(merged)

    def _evaluate_with_hint(
        self,
        condition_bool: torch.Tensor,
        captured: Sequence[torch.Tensor],
        use_then_branch: bool,
    ) -> Tuple[torch.Tensor, ...]:
        module = self.then_branch_module if use_then_branch else self.else_branch_module
        specs = self._then_arg_specs if use_then_branch else self._else_arg_specs
        return self._run_branch(module, specs, condition_bool, captured)

    def forward(  # type: ignore[override]
        self,
        condition: torch.Tensor,
        *captured: torch.Tensor,
    ):
        if torch.onnx.is_in_onnx_export():
            outputs = self._evaluate_without_cond(condition, captured)

            def _forward():
                if self._num_outputs == 0:
                    return ()
                if self._num_outputs == 1:
                    return outputs[0]
                return outputs

            attrs = self._onnx_attrs(get_onnx_version())
            result = IfExportToOnnx.export(
                _forward,
                "If",
                condition,
                attrs,
                self._num_outputs,
            )
        else:
            result = self._execute(condition, captured)

        if self._num_outputs == 0:
            return ()

        if isinstance(result, tuple):
            if self._num_outputs == 1:
                return result[0]
            return result

        return result

    def _onnx_attrs(self, opset_version: int) -> Dict[str, GraphProto]:
        del opset_version
        return {
            "then_branch": self._then_branch_proto,
            "else_branch": self._else_branch_proto,
        }


@add_converter(operation_type="If", version=1)
@add_converter(operation_type="If", version=11)
@add_converter(operation_type="If", version=13)
@add_converter(operation_type="If", version=16)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    if not node.input_values:
        raise RuntimeError("If node expects condition input")

    condition_input = node.input_values[0]
    condition_hint: Optional[bool] = None

    def _resolve_constant_scalar(value_name: str) -> Optional[float]:
        try:
            const_value = get_const_value(value_name, graph)
        except KeyError:
            pass
        else:
            if isinstance(const_value, torch.Tensor):
                if const_value.numel() == 1:
                    return float(const_value.item())
                return None
            if isinstance(const_value, (int, float, bool)):
                return float(const_value)
            if isinstance(const_value, (list, tuple)) and len(const_value) == 1:
                element = const_value[0]
                if isinstance(element, (int, float, bool)):
                    return float(element)
            return None

        value_type = graph.value_type(value_name)
        if value_type != ValueType.NODE_OUTPUT:
            return None

        producer, _ = graph.value_as_node_output(value_name)
        if not producer.input_values:
            return None

        if producer.operation_type in {
            "Cast",
            "Squeeze",
            "Unsqueeze",
            "Identity",
            "Reshape",
        }:
            return _resolve_constant_scalar(producer.input_values[0])

        return None

    scalar_value = _resolve_constant_scalar(condition_input)
    if scalar_value is not None:
        condition_hint = bool(scalar_value)

    then_branch_proto = None
    else_branch_proto = None
    for attribute in node.proto.attribute:
        if attribute.name == "then_branch":
            then_branch_proto = _copy_graph(attribute.g)
        elif attribute.name == "else_branch":
            else_branch_proto = _copy_graph(attribute.g)

    if then_branch_proto is None or else_branch_proto is None:
        raise RuntimeError(
            "If node attributes must include then_branch and else_branch"
        )

    then_module, then_inputs = _convert_branch(then_branch_proto, graph)
    else_module, else_inputs = _convert_branch(else_branch_proto, graph)

    captured_inputs: List[str] = []

    def _append_inputs(inputs: Sequence[str]) -> None:
        for name in inputs:
            if not name or name == condition_input:
                continue
            if name not in captured_inputs:
                captured_inputs.append(name)

    _append_inputs(then_inputs)
    _append_inputs(else_inputs)

    torch_module = OnnxIf(
        condition_input=condition_input,
        captured_inputs=captured_inputs,
        then_module=then_module,
        else_module=else_module,
        then_inputs=then_inputs,
        else_inputs=else_inputs,
        then_branch_proto=then_branch_proto,
        else_branch_proto=else_branch_proto,
        num_outputs=len(node.output_values),
        condition_hint=condition_hint,
    )

    onnx_mapping = OnnxMapping(
        inputs=tuple([condition_input, *captured_inputs]),
        outputs=node.output_values,
    )

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping,
    )
