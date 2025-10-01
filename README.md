onnx2torch is an ONNX to PyTorch converter.
Our converter:

- Is easy to use – Convert the ONNX model with the function call `convert`;
- Is easy to extend – Write your own custom layer in PyTorch and register it with `@add_converter`;
- Convert back to ONNX – You can convert the model back to ONNX using the `torch.onnx.export` function.

Please note that this converter covers only a limited number of PyTorch / ONNX models and operations.

## Installation

## create a new environment

```bash
# UPDATE IF NEEDED, OTHERWISE DELETE THIS COMMENT LINE
uv venv -p 3.12 .venv && source .venv/bin/activate
uv pip install -e .
```

## Usage

Below you can find some examples of use.

### Convert

```python
import onnx
import torch
from onnx2torch import convert

# Path to ONNX model
onnx_model_path = "/some/path/mobile_net_v2.onnx"
# You can pass the path to the onnx model to convert it or...
torch_model_1 = convert(onnx_model_path)

# Or you can load a regular onnx model and pass it to the converter
onnx_model = onnx.load(onnx_model_path)
torch_model_2 = convert(onnx_model)
```

### Execute

We can execute the returned `PyTorch model` in the same way as the original torch model.

```python
import onnxruntime as ort

# Create example data
x = torch.ones((1, 2, 224, 224)).cuda()

out_torch = torch_model_1(x)

ort_sess = ort.InferenceSession(onnx_model_path)
outputs_ort = ort_sess.run(None, {"input": x.numpy()})

# Check the Onnx output against PyTorch
print(torch.max(torch.abs(outputs_ort - out_torch.detach().numpy())))
print(np.allclose(outputs_ort, out_torch.detach().numpy(), atol=1.0e-7))
```

### CLI Conversion

The `run.py` helper script converts one or more ONNX models to `torch.export` bundles using a YAML configuration:

```bash
python run.py --cfg cfg/example.yml
```

When an ONNX model does not embed static input dimensions, list them under the `input_shapes` section of the config. Use the ONNX file path as the key (relative to the working directory) and provide per-input shapes, for example:

```yaml
input_shapes:
  data/onnx/model_without_shapes.onnx:
    input_1: [1, 3, 224, 224]
```

The converter validates that every shape-less model referenced in the config supplies these overrides before running [`onnxsim.simplify`](https://github.com/daquexian/onnx-simplifier) to bake the dimensions into the graph.

## How to add new operations to converter

### :page_facing_up: List of currently supported operations can be founded [here](operators.md).

Here we show how to extend onnx2torch with new ONNX operation, that supported by both PyTorch and ONNX

<details>
<summary>and has the same behaviour</summary>

An example of such a module is [Relu](./onnx2torch/node_converters/activations.py)

```python
@add_converter(operation_type="Relu", version=6)
@add_converter(operation_type="Relu", version=13)
@add_converter(operation_type="Relu", version=14)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    return OperationConverterResult(
        torch_module=nn.ReLU(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
```

Here we have registered an operation named `Relu` for opset versions 6, 13, 14.
Note that the `torch_module` argument in `OperationConverterResult` must be a torch.nn.Module, not just a callable object!
If Operation's behaviour differs from one opset version to another, you should implement it separately.

</details>

<details>
<summary>but has different behaviour</summary>

An example of such a module is [ScatterND](./onnx2torch/node_converters/scatter_nd.py)

```python
# It is recommended to use Enum for string ONNX attributes.
class ReductionOnnxAttr(Enum):
    NONE = "none"
    ADD = "add"
    MUL = "mul"


class OnnxScatterND(nn.Module, OnnxToTorchModuleWithCustomExport):
    def __init__(self, reduction: ReductionOnnxAttr):
        super().__init__()
        self._reduction = reduction

    # The following method should return ONNX attributes with their values as a dictionary.
    # The number of attributes, their names and values depend on opset version;
    # method should return correct set of attributes.
    # Note: add type-postfix for each key: reduction -> reduction_s, where s means "string".
    def _onnx_attrs(self, opset_version: int) -> Dict[str, Any]:
        onnx_attrs: Dict[str, Any] = {}

        # Here we handle opset versions < 16 where there is no "reduction" attribute.
        if opset_version < 16:
            if self._reduction != ReductionOnnxAttr.NONE:
                raise ValueError(
                    "ScatterND from opset < 16 does not support"
                    f"reduction attribute != {ReductionOnnxAttr.NONE.value},"
                    f"got {self._reduction.value}"
                )
            return onnx_attrs

        onnx_attrs["reduction_s"] = self._reduction.value
        return onnx_attrs

    def forward(
        self,
        data: torch.Tensor,
        indices: torch.Tensor,
        updates: torch.Tensor,
    ) -> torch.Tensor:
        def _forward():
            # ScatterND forward implementation...
            return output

        if torch.onnx.is_in_onnx_export():
            # Please follow our convention, args consists of:
            # forward function, operation type, operation inputs, operation attributes.
            onnx_attrs = self._onnx_attrs(opset_version=get_onnx_version())
            return DefaultExportToOnnx.export(
                _forward, "ScatterND", data, indices, updates, onnx_attrs
            )

        return _forward()


@add_converter(operation_type="ScatterND", version=11)
@add_converter(operation_type="ScatterND", version=13)
@add_converter(operation_type="ScatterND", version=16)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    node_attributes = node.attributes
    reduction = ReductionOnnxAttr(node_attributes.get("reduction", "none"))
    return OperationConverterResult(
        torch_module=OnnxScatterND(reduction=reduction),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
```

Here we have used a trick to convert the model from torch back to ONNX by defining the custom `_ScatterNDExportToOnnx`.

</details>

## Opset version workaround

Incase you are using a model with older opset, try the following workaround:

[ONNX Version Conversion - Official Docs](https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#converting-version-of-an-onnx-model-within-default-domain-aionnx)

<details>
<summary>Example</summary>

```python
import onnx
from onnx import version_converter
import torch
from onnx2torch import convert

# Load the ONNX model.
model = onnx.load("model.onnx")
# Convert the model to the target version.
target_version = 13
converted_model = version_converter.convert_version(model, target_version)
# Convert to torch.
torch_model = convert(converted_model)
torch.save(torch_model, "model.pt")
```

</details>

Note: use this only when the model does not convert to PyTorch using the existing opset version. Result might vary.


## Acknowledgments

This repo is based on https://github.com/ENOT-AutoDL/onnx2torch
