from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import onnx
import pytest
import torch

from onnx2torch.converter import convert
from onnx2torch.node_converters.trilu import OnnxTrilu

try:
    from tests.utils.common import make_model_from_nodes
except ImportError:  # pragma: no cover - allow running tests without package alias
    from unit_tests.utils.common import make_model_from_nodes  # type: ignore[no-redef]

try:  # pragma: no cover - FakeTensor may be unavailable on some torch builds
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
except ImportError:  # pragma: no cover - skip fake tensor tests when unsupported
    FakeTensorMode = None  # type: ignore[assignment]
    ShapeEnv = None  # type: ignore[assignment]

try:  # pragma: no cover - torch.export may be unavailable on some torch releases
    from torch.export import Dim
    from torch.export import export as export_program
except ImportError:  # pragma: no cover - skip export tests when unsupported
    Dim = None  # type: ignore[assignment]
    export_program = None  # type: ignore[assignment]


def _build_trilu_model(
    data: np.ndarray,
    upper: bool,
    k_value: Optional[int],
    k_as_initializer: bool,
) -> Tuple[onnx.ModelProto, Dict[str, np.ndarray]]:
    test_inputs: Dict[str, np.ndarray] = {"x": data}
    node_inputs = ["x"]
    initializers: Dict[str, np.ndarray] = {}

    if k_value is not None:
        k_array = np.array(k_value, dtype=np.int64)
        if k_as_initializer:
            node_inputs.append("k_init")
            initializers["k_init"] = k_array
        else:
            node_inputs.append("k")
            test_inputs["k"] = k_array

    node = onnx.helper.make_node(
        op_type="Trilu",
        inputs=node_inputs,
        outputs=["y"],
        upper=int(upper),
    )
    model = make_model_from_nodes(
        nodes=node,
        initializers=initializers,
        inputs_example=test_inputs,
        outputs_info=None,
        opset_version=14,
    )
    return model, test_inputs


@pytest.mark.parametrize(
    ("upper", "k_value", "k_as_initializer"),
    (
        (False, None, False),
        (True, 0, True),
        (False, -1, True),
        (True, 2, False),
    ),
)
def test_trilu_converter_matches_ort(
    upper: bool,
    k_value: Optional[int],
    k_as_initializer: bool,
) -> None:
    data = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    model, test_inputs = _build_trilu_model(
        data=data,
        upper=upper,
        k_value=k_value,
        k_as_initializer=k_as_initializer,
    )
    torch_model = convert(model).eval()

    torch_inputs = []
    for value_info in model.graph.input:
        np_value = test_inputs[value_info.name]
        tensor = torch.from_numpy(np_value)
        torch_inputs.append(tensor)

    outputs = torch_model(*torch_inputs)
    output_tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

    input_tensor = torch_inputs[0]
    if len(torch_inputs) > 1:
        diagonal = int(torch_inputs[1].item())
    elif k_value is not None:
        diagonal = int(k_value)
    else:
        diagonal = 0

    expected = (
        torch.triu(input_tensor, diagonal=diagonal)
        if upper
        else torch.tril(input_tensor, diagonal=diagonal)
    )

    torch.testing.assert_close(output_tensor, expected)


@pytest.mark.parametrize(
    ("upper", "k_value"),
    (
        (False, None),
        (False, -2),
        (True, 1),
    ),
)
def test_trilu_module_matches_torch(
    upper: bool,
    k_value: Optional[int],
) -> None:
    module = OnnxTrilu(upper=upper)
    input_tensor = torch.randn(3, 5, dtype=torch.float32)

    if k_value is None:
        result = module(input_tensor)
        diagonal = 0
    else:
        if k_value >= 0:
            result = module(input_tensor, k_value)
        else:
            diagonal_tensor = torch.tensor(k_value, dtype=torch.int64)
            result = module(input_tensor, diagonal_tensor)
        diagonal = int(k_value)

    expected = (
        torch.triu(input_tensor, diagonal=diagonal)
        if upper
        else torch.tril(input_tensor, diagonal=diagonal)
    )
    torch.testing.assert_close(result, expected)


@pytest.mark.skipif(FakeTensorMode is None, reason="FakeTensorMode is unavailable")
def test_trilu_accepts_fake_tensor_inputs() -> None:
    assert FakeTensorMode is not None  # guarded by skipif
    assert ShapeEnv is not None

    module = OnnxTrilu(upper=False)
    mode = FakeTensorMode(shape_env=ShapeEnv())
    with mode:
        input_tensor = torch.randn(2, 3)
        diagonal = torch.tensor(1, dtype=torch.int64)
        _ = module(input_tensor, diagonal)

    real_input = torch.randn(2, 3)
    diagonal_real = torch.tensor(1, dtype=torch.int64)
    real_out = module(real_input, diagonal_real)
    expected = torch.tril(real_input, diagonal=int(diagonal_real.item()))
    torch.testing.assert_close(real_out, expected)


@pytest.mark.skipif(export_program is None, reason="torch.export is unavailable")
def test_trilu_traces_with_optional_diagonal() -> None:
    class ExportModel(torch.nn.Module):
        def __init__(self, upper: bool) -> None:
            super().__init__()
            self._op = OnnxTrilu(upper=upper)

        def forward(
            self,
            input_tensor: torch.Tensor,
            diagonal: torch.Tensor,
        ) -> torch.Tensor:  # type: ignore[override]
            return self._op(input_tensor, diagonal)

    model = ExportModel(upper=True).eval()
    sample = torch.randn(3, 3)
    diagonal = torch.tensor(1, dtype=torch.int64)

    assert export_program is not None
    assert Dim is not None

    dynamic_shapes = (
        {0: Dim("row"), 1: Dim("col")},
        {},
    )

    exported = export_program(
        model,
        (sample, diagonal),
        dynamic_shapes=dynamic_shapes,
    )
    assert exported is not None
