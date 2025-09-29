#!/usr/bin/env python3
"""CLI wrapper to convert ONNX models to torch.export programs."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import torch
import yaml
from onnx.onnx_ml_pb2 import ModelProto, ValueInfoProto

from onnx2torch.converter import convert
from onnx2torch.utils.dtype import onnx_dtype_to_torch_dtype
from onnx2torch.utils.safe_shape_inference import safe_shape_inference

try:  # torch < 2.1 does not expose torch.export.export
    from torch.export import export as export_program
except (
    ImportError,
    AttributeError,
) as exc:  # pragma: no cover - guard for older torch versions
    raise RuntimeError(
        "torch>=2.1 is required to export models to .pt2 format"
    ) from exc

LOGGER = logging.getLogger("onnx2torch.runner")
DEFAULT_OUTPUT_DIR = Path("data/executorch")
SUPPORTED_FILLS = {"zeros", "ones", "random"}


@dataclass
class InputConfig:
    path: Path
    pattern: str = "*.onnx"
    recursive: bool = True


@dataclass
class ExampleInputConfig:
    default_fill: str = "zeros"
    overrides: Dict[str, Dict[str, object]] = field(default_factory=dict)


@dataclass
class RunnerConfig:
    inputs: List[InputConfig]
    output_dir: Path = DEFAULT_OUTPUT_DIR
    save_input_names: bool = False
    attach_onnx_mapping: bool = False
    device: str = "cpu"
    example_inputs: ExampleInputConfig = field(default_factory=ExampleInputConfig)


@dataclass
class ModelTask:
    source: Path
    destination: Path


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING - min(verbosity, 2) * 10
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _load_config(path: Path) -> RunnerConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}

    inputs_raw = data.get("inputs", data.get("input"))
    if inputs_raw is None:
        raise ValueError('Config must define "inputs" (string, dict, or list).')

    if isinstance(inputs_raw, (str, Path)):
        inputs_raw = [{"path": inputs_raw}]
    elif isinstance(inputs_raw, dict):
        inputs_raw = [inputs_raw]
    elif not isinstance(inputs_raw, list):
        raise TypeError(
            'The "inputs" entry must be a string, dict, or list of dicts/strings.'
        )

    inputs: List[InputConfig] = []
    for entry in inputs_raw:
        if isinstance(entry, (str, Path)):
            inputs.append(InputConfig(path=Path(entry)))
            continue

        if not isinstance(entry, dict):
            raise TypeError(
                'Each "inputs" item must be either a string path or a mapping.'
            )

        if "path" not in entry:
            raise ValueError('Missing "path" in inputs item.')

        inputs.append(
            InputConfig(
                path=Path(entry["path"]),
                pattern=entry.get("pattern", "*.onnx"),
                recursive=entry.get("recursive", True),
            )
        )

    output_dir = Path(data.get("output_dir", DEFAULT_OUTPUT_DIR))

    conversion_cfg = data.get("conversion", {})
    if not isinstance(conversion_cfg, dict):
        raise TypeError('The "conversion" section must be a mapping if provided.')

    example_cfg_raw = data.get("example_inputs", {})
    if isinstance(example_cfg_raw, dict):
        default_fill = example_cfg_raw.get("default_fill", "zeros")
        overrides = example_cfg_raw.get("overrides", example_cfg_raw.get("inputs", {}))
    elif example_cfg_raw in (None, ""):
        default_fill = "zeros"
        overrides = {}
    else:
        raise TypeError('The "example_inputs" section must be a mapping if provided.')

    example_cfg = ExampleInputConfig(
        default_fill=str(default_fill).lower(),
        overrides={key: dict(value) for key, value in (overrides or {}).items()},
    )

    if example_cfg.default_fill not in SUPPORTED_FILLS:
        raise ValueError(
            f'"default_fill" must be one of {sorted(SUPPORTED_FILLS)}, got {example_cfg.default_fill!r}.',
        )

    device = data.get("device", "cpu")

    return RunnerConfig(
        inputs=inputs,
        output_dir=output_dir,
        save_input_names=bool(conversion_cfg.get("save_input_names", False)),
        attach_onnx_mapping=bool(conversion_cfg.get("attach_onnx_mapping", False)),
        device=str(device),
        example_inputs=example_cfg,
    )


def _iter_model_paths(input_cfg: InputConfig) -> Iterator[Path]:
    root = input_cfg.path.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")

    if root.is_file():
        if root.suffix.lower() != ".onnx":
            raise ValueError(f"Expected an ONNX file, got: {root}")
        yield root
        return

    iterator = (
        root.rglob(input_cfg.pattern)
        if input_cfg.recursive
        else root.glob(input_cfg.pattern)
    )
    for candidate in iterator:
        if candidate.is_file() and candidate.suffix.lower() == ".onnx":
            yield candidate


def _collect_tasks(config: RunnerConfig) -> List[ModelTask]:
    tasks: List[ModelTask] = []
    seen: Dict[Path, Path] = {}

    for entry in config.inputs:
        entry_root = entry.path.expanduser().resolve()
        for model_path in _iter_model_paths(entry):
            if model_path in seen:
                LOGGER.debug("Skipping duplicate model path: %s", model_path)
                continue

            if entry_root.is_file():
                relative = model_path.name
            else:
                try:
                    relative = model_path.relative_to(entry_root)
                except ValueError:
                    relative = model_path.name

            destination = config.output_dir.expanduser() / Path(relative).with_suffix(
                ".pt2"
            )
            tasks.append(ModelTask(source=model_path, destination=destination))
            seen[model_path] = destination

    return tasks


def _dim_sizes(value_info: ValueInfoProto) -> List[int]:
    dims: List[int] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value") and dim.dim_value > 0:
            dims.append(dim.dim_value)
        else:
            dims.append(1)
    return dims


def _resolve_dtype(
    name: str, elem_type: int, override: Dict[str, object]
) -> torch.dtype:
    dtype_override = override.get("dtype")
    if dtype_override is not None:
        torch_dtype = _dtype_from_string(str(dtype_override))
        LOGGER.debug("Using dtype override for %s: %s", name, torch_dtype)
        return torch_dtype

    torch_dtype = onnx_dtype_to_torch_dtype(elem_type)
    if torch_dtype in (str, bool):
        raise ValueError(f"Input {name} has unsupported dtype {torch_dtype}.")
    return torch_dtype


def _dtype_from_string(value: str) -> torch.dtype:
    key = value.strip().lower()
    aliases = {
        "float": torch.float32,
        "float32": torch.float32,
        "fp32": torch.float32,
        "half": torch.float16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "double": torch.float64,
        "float64": torch.float64,
        "int": torch.int32,
        "int32": torch.int32,
        "int64": torch.int64,
        "long": torch.int64,
        "int16": torch.int16,
        "short": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    if key not in aliases:
        raise ValueError(f"Unsupported dtype override: {value}")
    return aliases[key]


def _create_example_tensor(
    name: str,
    value_info: ValueInfoProto,
    override: Dict[str, object],
    default_fill: str,
    device: str,
) -> torch.Tensor:
    if "shape" in override:
        shape = list(int(size) for size in override["shape"])
    else:
        shape = _dim_sizes(value_info)
        if override.get("ensure_min_dim"):
            min_dim = int(override["ensure_min_dim"])
            shape = [max(dim, min_dim) for dim in shape]

    if not shape:
        shape = []

    torch_dtype = _resolve_dtype(name, value_info.type.tensor_type.elem_type, override)

    fill = str(override.get("fill", default_fill)).lower()
    if fill not in SUPPORTED_FILLS:
        raise ValueError(f"Unsupported fill strategy {fill!r} for input {name}.")

    shape_tuple: Tuple[int, ...] = tuple(int(max(1, dim)) for dim in shape)

    if fill == "zeros":
        return torch.zeros(shape_tuple, dtype=torch_dtype, device=device)
    if fill == "ones":
        return torch.ones(shape_tuple, dtype=torch_dtype, device=device)

    # fill == 'random'
    if torch_dtype.is_floating_point or torch_dtype.is_complex:
        return torch.randn(shape_tuple, dtype=torch_dtype, device=device)
    if torch_dtype == torch.bool:
        return torch.randint(0, 2, shape_tuple, dtype=torch.bool, device=device)
    # Integers default to uniform range [0, 10).
    return torch.randint(0, 10, shape_tuple, dtype=torch_dtype, device=device)


def _build_example_args(
    model: ModelProto, config: RunnerConfig
) -> Tuple[torch.Tensor, ...]:
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    overrides = config.example_inputs.overrides
    tensors: List[torch.Tensor] = []

    for value_info in model.graph.input:
        if value_info.name in initializer_names:
            continue

        override = overrides.get(value_info.name, {})
        tensor = _create_example_tensor(
            name=value_info.name,
            value_info=value_info,
            override=override,
            default_fill=config.example_inputs.default_fill,
            device=config.device,
        )
        tensors.append(tensor)

    return tuple(tensors)


def _export_model(task: ModelTask, config: RunnerConfig) -> None:
    LOGGER.info("Converting %s", task.source)
    model_with_shapes = safe_shape_inference(task.source)

    with torch.inference_mode():
        module = convert(
            task.source,
            save_input_names=config.save_input_names,
            attach_onnx_mapping=config.attach_onnx_mapping,
        )
        module.eval()
        module.to(config.device)

        example_args = _build_example_args(model_with_shapes, config)
        LOGGER.debug(
            "Example args for %s: %s",
            task.source,
            [tuple(t.shape) for t in example_args],
        )

        exported = export_program(module, example_args)

    task.destination.parent.mkdir(parents=True, exist_ok=True)
    exported.save(str(task.destination))
    LOGGER.info("Saved %s", task.destination)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert ONNX models to .pt2 (ExportedProgram)."
    )
    parser.add_argument(
        "--cfg", required=True, type=Path, help="Path to the YAML config file."
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase logging verbosity."
    )

    args = parser.parse_args(argv)

    _setup_logging(args.verbose)

    config = _load_config(args.cfg)
    tasks = _collect_tasks(config)

    if not tasks:
        LOGGER.warning("No ONNX models found. Nothing to do.")
        return 0

    for task in tasks:
        _export_model(task, config)

    LOGGER.info("Converted %d model(s).", len(tasks))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
