import pytest
import torch

from run import _strip_runtime_guards


def test_strip_runtime_guards_removes_scalar_asserts() -> None:
    graph = torch.fx.Graph()
    tensor_in = graph.placeholder("x")
    item = graph.call_function(torch.ops.aten.item.default, (tensor_in,))
    graph.call_function(torch.ops.aten._assert_scalar.default, (item,))
    graph.output((tensor_in,))

    module = torch.fx.GraphModule(torch.nn.Module(), graph)

    class DummyExport:
        def __init__(self, graph_module: torch.fx.GraphModule) -> None:
            self.graph_module = graph_module

    exported = DummyExport(module)

    assert any(
        node.target is torch.ops.aten._assert_scalar.default
        for node in exported.graph_module.graph.nodes
    )

    _strip_runtime_guards(exported)

    assert all(
        node.target is not torch.ops.aten._assert_scalar.default
        for node in exported.graph_module.graph.nodes
    )
    assert all(
        node.target is not torch.ops.aten.item.default
        for node in exported.graph_module.graph.nodes
    )


def test_strip_runtime_guards_removes_sym_constrain() -> None:
    graph = torch.fx.Graph()
    tensor_in = graph.placeholder("x")
    item = graph.call_function(torch.ops.aten.item.default, (tensor_in,))
    graph.call_function(
        torch.ops.aten.sym_constrain_range_for_size.default,
        (item,),
        {"min": 1, "max": 64},
    )
    graph.output((tensor_in,))

    module = torch.fx.GraphModule(torch.nn.Module(), graph)

    class DummyExport:
        def __init__(self, graph_module: torch.fx.GraphModule) -> None:
            self.graph_module = graph_module

    exported = DummyExport(module)

    assert any(
        node.target is torch.ops.aten.sym_constrain_range_for_size.default
        for node in exported.graph_module.graph.nodes
    )

    _strip_runtime_guards(exported)

    assert all(
        node.target is not torch.ops.aten.sym_constrain_range_for_size.default
        for node in exported.graph_module.graph.nodes
    )
    assert all(
        node.target is not torch.ops.aten.item.default
        for node in exported.graph_module.graph.nodes
    )


def test_strip_runtime_guards_removes_tensor_metadata_asserts() -> None:
    tensor_metadata_guard = getattr(torch.ops.aten, "_assert_tensor_metadata", None)
    if tensor_metadata_guard is None:
        pytest.skip("torch build does not expose aten._assert_tensor_metadata")

    graph = torch.fx.Graph()
    tensor_in = graph.placeholder("x")
    reshaped = graph.call_function(
        torch.ops.aten.reshape.default,
        (tensor_in, [1, 8]),
    )
    graph.call_function(tensor_metadata_guard.default, (reshaped,))
    graph.output((tensor_in,))

    module = torch.fx.GraphModule(torch.nn.Module(), graph)

    class DummyExport:
        def __init__(self, graph_module: torch.fx.GraphModule) -> None:
            self.graph_module = graph_module

    exported = DummyExport(module)

    assert any(
        node.target is tensor_metadata_guard.default
        for node in exported.graph_module.graph.nodes
    )

    _strip_runtime_guards(exported)

    assert all(
        node.target is not tensor_metadata_guard.default
        for node in exported.graph_module.graph.nodes
    )
    assert all(
        node.target is not torch.ops.aten.reshape.default
        for node in exported.graph_module.graph.nodes
    )
