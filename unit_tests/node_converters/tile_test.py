import numpy as np
import onnx
import pytest
import torch
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from onnx2torch.node_converters.tile import OnnxTile
from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes

try:  # pragma: no cover - FakeTensor may be unavailable on some torch builds
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
except ImportError:  # pragma: no cover - keep skipif working without FakeTensor
    FakeTensorMode = None  # type: ignore[assignment]
    ShapeEnv = None  # type: ignore[assignment]

try:  # pragma: no cover - torch.export may not exist on some torch releases
    from torch.export import Dim
    from torch.export import export as export_program
except ImportError:  # pragma: no cover - skip export-specific tests when unavailable
    Dim = None  # type: ignore[assignment]
    export_program = None  # type: ignore[assignment]


def _test_tile(
    data: np.ndarray,
    repeats: np.ndarray,
    desire_out: np.ndarray,
) -> None:
    test_inputs = {"input_tensor": data, "repeats": repeats}
    node = onnx.helper.make_node(
        op_type="Tile",
        inputs=list(test_inputs),
        outputs=["y"],
    )
    outputs_info = [
        make_tensor_value_info(
            name="y",
            elem_type=NP_TYPE_TO_TENSOR_TYPE[data.dtype],
            shape=desire_out.shape,
        ),
    ]
    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
        outputs_info=outputs_info,
    )
    check_onnx_model(model, test_inputs)


@pytest.mark.filterwarnings("ignore::torch.jit._trace.TracerWarning")
def test_tile() -> None:  # pylint: disable=missing-function-docstring
    data = np.random.rand(2, 3, 4, 5).astype(np.float32)
    repeats = np.random.randint(low=1, high=10, size=(np.ndim(data),)).astype(np.int64)
    _test_tile(
        data=data,
        repeats=repeats,
        desire_out=np.tile(data, repeats),
    )

    data = np.array([[0, 1], [2, 3]], dtype=np.float32)

    repeats = np.array([2, 2], dtype=np.int64)
    _test_tile(
        data=data,
        repeats=repeats,
        desire_out=np.array(
            [[0, 1, 0, 1], [2, 3, 2, 3], [0, 1, 0, 1], [2, 3, 2, 3]], dtype=np.float32
        ),
    )


@pytest.mark.skipif(FakeTensorMode is None, reason="FakeTensorMode is unavailable")
def test_tile_accepts_fake_tensor_repeats() -> None:  # pylint: disable=missing-function-docstring
    module = OnnxTile()

    assert ShapeEnv is not None  # for mypy; guarded by skipif
    mode = FakeTensorMode(shape_env=ShapeEnv())
    with mode:
        fake_input = torch.randn(2, 3)
        repeat_dims = [
            torch.ops.aten.scalar_tensor.default(
                torch.ops.aten.sym_size.int(fake_input, 0),
                dtype=torch.int64,
                device=fake_input.device,
            ),
            torch.tensor(1, dtype=torch.int64, device=fake_input.device),
        ]
        repeats = torch.ops.aten.stack.default(repeat_dims, 0)
        result = module(fake_input, repeats)
        assert isinstance(result, torch.Tensor)

    real_result = module(
        torch.randn(2, 3), torch.tensor([2, 1], dtype=torch.int64)
    )
    assert real_result.shape == (4, 3)


@pytest.mark.skipif(export_program is None, reason="torch.export is unavailable")
def test_tile_traces_with_symbolic_repeats() -> None:  # pylint: disable=missing-function-docstring
    class ExportModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tile = OnnxTile()

        def forward(  # type: ignore[override]
            self, input_tensor: torch.Tensor
        ) -> torch.Tensor:
            repeats = torch.ops.aten.stack.default(
                [
                    torch.ops.aten.scalar_tensor.default(
                        torch.ops.aten.sym_size.int(input_tensor, 0),
                        dtype=torch.int64,
                        device=input_tensor.device,
                    ),
                    torch.tensor(1, dtype=torch.int64, device=input_tensor.device),
                ],
                0,
            )
            return self.tile(input_tensor, repeats)

    model = ExportModel().eval()
    sample = torch.randn(2, 3)

    assert Dim is not None  # mypy hint; guarded by skip
    dynamic_shapes = ({0: Dim("batch"), 1: Dim("feature")},)

    exported = export_program(model, (sample,), dynamic_shapes=dynamic_shapes)
    assert exported is not None
