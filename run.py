#!/usr/bin/env python3
"""CLI wrapper to convert ONNX models to torch.export programs."""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import shutil
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Sequence, Set, Tuple

import torch
import yaml
import onnx
from onnx.onnx_ml_pb2 import ModelProto, TensorProto, ValueInfoProto

from onnx2torch.converter import convert
from onnx2torch.utils.dtype import onnx_dtype_to_torch_dtype
from onnx2torch.utils.safe_shape_inference import safe_shape_inference

try:  # noqa: SIM105 - onnxsim may attempt to auto-install onnxruntime at import time
    from onnxsim import simplify as _onnxsim_simplify
except Exception as exc:  # noqa: BLE001 - propagate detailed failure later
    _ONNXSIM_IMPORT_ERROR: Exception | None = exc
    _onnxsim_simplify = None
else:
    _ONNXSIM_IMPORT_ERROR = None

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
RUN_LOG_ROOT = Path("logs/run_logs")
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
    input_shapes: Dict[Path, Dict[str, Tuple[int, ...]]] = field(default_factory=dict)


@dataclass
class ModelTask:
    source: Path
    destination: Path


@dataclass
class RunContext:
    directory: Path
    verbosity: int

    @property
    def log_file(self) -> Path:
        return self.directory / "run.log"

    @property
    def show_full_trace(self) -> bool:
        return self.verbosity >= 2


class ExportError(RuntimeError):
    """Raised when exporting a model fails without emitting verbose tracebacks."""

    def __init__(
        self, source: Path, detail_path: Path, message: str | None = None
    ) -> None:
        detail_msg = message or "Failed to export model."
        super().__init__(f"{detail_msg} See {detail_path} for details.")
        self.source = source
        self.detail_path = detail_path


class ShapePreparationError(RuntimeError):
    """Raised when a model lacks required static input shape information."""


def _prepare_run_directory() -> Path:
    RUN_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    candidate = RUN_LOG_ROOT / timestamp
    suffix = 1
    while candidate.exists():
        candidate = RUN_LOG_ROOT / f"{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _create_run_context(verbosity: int) -> RunContext:
    directory = _prepare_run_directory()
    return RunContext(directory=directory, verbosity=verbosity)


def _setup_logging(run_context: RunContext) -> None:
    console_level = logging.WARNING - min(run_context.verbosity, 2) * 10
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    file_handler = logging.FileHandler(run_context.log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    )

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


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

    input_shapes_raw = data.get("input_shapes", {}) or {}
    if not isinstance(input_shapes_raw, dict):
        raise TypeError('The "input_shapes" section must be a mapping if provided.')

    input_shape_overrides: Dict[Path, Dict[str, Tuple[int, ...]]] = {}
    for model_key, shapes in input_shapes_raw.items():
        if not isinstance(shapes, dict):
            raise TypeError(
                f'Each entry under "input_shapes" must be a mapping of input names to shapes; got {type(shapes).__name__!r} for {model_key!r}.',
            )

        resolved_model_path = Path(model_key).expanduser().resolve()
        if resolved_model_path in input_shape_overrides:
            raise ValueError(
                f"Duplicate input_shapes entry for {resolved_model_path}.",
            )

        override: Dict[str, Tuple[int, ...]] = {}
        for input_name, dims in shapes.items():
            if isinstance(dims, dict):
                if "shape" not in dims:
                    raise TypeError(
                        f'Input shape override for {input_name!r} in {model_key!r} must include a "shape" key.',
                    )
                dims = dims["shape"]

            if not isinstance(dims, (list, tuple)):
                raise TypeError(
                    f"Input shape override for {input_name!r} in {model_key!r} must be a sequence of integers.",
                )

            try:
                dims_tuple = tuple(int(dim) for dim in dims)
            except (TypeError, ValueError) as exc:  # noqa: PERF203 - explicit error handling improves messaging
                raise TypeError(
                    f"Input shape override for {input_name!r} in {model_key!r} must contain only integers.",
                ) from exc

            if not dims_tuple:
                raise ValueError(
                    f"Input shape override for {input_name!r} in {model_key!r} cannot be empty.",
                )

            if any(dimension <= 0 for dimension in dims_tuple):
                raise ValueError(
                    f"Input shape override for {input_name!r} in {model_key!r} must contain positive dimensions.",
                )

            override[str(input_name)] = dims_tuple

        input_shape_overrides[resolved_model_path] = override

    device = data.get("device", "cpu")

    return RunnerConfig(
        inputs=inputs,
        output_dir=output_dir,
        save_input_names=bool(conversion_cfg.get("save_input_names", False)),
        attach_onnx_mapping=bool(conversion_cfg.get("attach_onnx_mapping", False)),
        device=str(device),
        example_inputs=example_cfg,
        input_shapes=input_shape_overrides,
    )


def _persist_run_config(cfg_path: Path, run_context: RunContext) -> None:
    destination = run_context.directory / cfg_path.name
    try:
        shutil.copy2(cfg_path, destination)
    except FileNotFoundError:
        LOGGER.warning("Unable to copy config file %s into run directory.", cfg_path)


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


def _input_names_missing_static_shapes(model: ModelProto) -> Set[str]:
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    missing: Set[str] = set()

    for value_info in model.graph.input:
        if value_info.name in initializer_names:
            continue

        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            missing.add(value_info.name)
            continue

        dims = tensor_type.shape.dim
        if not dims:
            continue

        for dim in dims:
            if not dim.HasField("dim_value") or dim.dim_value <= 0:
                missing.add(value_info.name)
                break

    return missing


def _lookup_input_shape_overrides(
    config: RunnerConfig, model_path: Path
) -> Dict[str, Tuple[int, ...]]:
    resolved_path = model_path.expanduser().resolve()
    return config.input_shapes.get(resolved_path, {})


def _clone_model(model: ModelProto) -> ModelProto:
    clone = ModelProto()
    clone.CopyFrom(model)
    return clone


def _apply_input_shape_overrides_without_onnxsim(
    model: ModelProto,
    overrides: Mapping[str, Sequence[int]],
    source: Path,
) -> ModelProto:
    patched_model = _clone_model(model)
    inputs_by_name = {
        value_info.name: value_info for value_info in patched_model.graph.input
    }

    unknown_overrides = set(overrides).difference(inputs_by_name)
    if unknown_overrides:
        raise ShapePreparationError(
            "Shape overrides provided for unknown inputs {inputs} in model {source}.".format(
                inputs=", ".join(sorted(unknown_overrides)), source=source
            )
        )

    for name, dims in overrides.items():
        value_info = inputs_by_name[name]
        if value_info.type.WhichOneof("value") != "tensor_type":
            raise ShapePreparationError(
                f"Input {name} in model {source} is not a tensor; cannot override its shape."
            )

        tensor_type = value_info.type.tensor_type
        shape = tensor_type.shape
        shape.dim.clear()
        for dimension in dims:
            dim_proto = shape.dim.add()
            dim_proto.dim_value = int(dimension)

    return patched_model


def _restore_initializer_data_from_source(
    target: ModelProto,
    initializer_store: Mapping[str, TensorProto],
    model_path: Path,
) -> None:
    missing_sources = []

    for initializer in target.graph.initializer:
        if initializer.raw_data:
            continue

        original = initializer_store.get(initializer.name)
        if original is None:
            missing_sources.append(initializer.name)
            continue

        initializer.CopyFrom(original)

    if missing_sources:
        raise ShapePreparationError(
            "Unable to restore tensor data for inputs {inputs} in model {source}.".format(
                inputs=", ".join(sorted(missing_sources)), source=model_path
            )
        )


def _prepare_model_with_shapes(task: ModelTask, config: RunnerConfig) -> ModelProto:
    try:
        raw_model = onnx.load(str(task.source))
    except Exception as exc:  # noqa: BLE001 - propagate context to user
        raise ShapePreparationError(
            f"Failed to load ONNX model {task.source}: {exc}"
        ) from exc

    initializer_store: Dict[str, TensorProto] = {}
    for initializer in raw_model.graph.initializer:
        clone = TensorProto()
        clone.CopyFrom(initializer)
        initializer_store[initializer.name] = clone

    try:
        inferred_model = safe_shape_inference(raw_model)
    except Exception as exc:  # noqa: BLE001 - inference can fail when shapes are missing
        LOGGER.debug("Initial shape inference failed for %s: %s", task.source, exc)
        inferred_model = raw_model

    missing_inputs = _input_names_missing_static_shapes(inferred_model)
    if not missing_inputs:
        LOGGER.debug("Model %s already has static input shapes.", task.source)
        return inferred_model

    overrides = _lookup_input_shape_overrides(config, task.source)
    if not overrides:
        formatted = ", ".join(sorted(missing_inputs)) or "unknown inputs"
        raise ShapePreparationError(
            "Model {source} has dynamic or missing shapes for inputs: {inputs}. "
            "Provide static dimensions under the config's `input_shapes` section.".format(
                source=task.source, inputs=formatted
            )
        )

    missing_overrides = missing_inputs.difference(overrides.keys())
    if missing_overrides:
        raise ShapePreparationError(
            "Missing shape overrides for inputs {inputs} in model {source}. "
            "Update the `input_shapes` section to include them.".format(
                inputs=", ".join(sorted(missing_overrides)), source=task.source
            )
        )

    override_payload = {
        name: [int(dimension) for dimension in dimensions]
        for name, dimensions in overrides.items()
    }

    LOGGER.debug(
        "Applying input shape overrides for %s: %s",
        task.source,
        {name: tuple(dims) for name, dims in override_payload.items()},
    )

    def apply_overrides_manually() -> ModelProto:
        try:
            patched = _apply_input_shape_overrides_without_onnxsim(
                raw_model, override_payload, task.source
            )
        except ShapePreparationError:
            raise
        except Exception as manual_exc:  # noqa: BLE001 - propagate context to user
            raise ShapePreparationError(
                f"Failed to apply shape overrides for {task.source}: {manual_exc}"
            ) from manual_exc

        try:
            inferred = safe_shape_inference(patched)
        except Exception as inference_exc:  # noqa: BLE001 - inference can still fail
            raise ShapePreparationError(
                "Shape inference failed after applying overrides for {source}: {error}".format(
                    source=task.source, error=inference_exc
                )
            ) from inference_exc

        if any(not initializer.raw_data for initializer in inferred.graph.initializer):
            _restore_initializer_data_from_source(
                inferred, initializer_store, task.source
            )

        return inferred

    if _onnxsim_simplify is None:
        LOGGER.info(
            "onnxsim is unavailable; applying shape overrides directly for %s.",
            task.source,
        )
        return apply_overrides_manually()

    try:
        simplified_model, ok = _onnxsim_simplify(
            model=str(task.source),
            overwrite_input_shapes=override_payload,
        )
    except Exception as exc:  # noqa: BLE001 - onnxsim raises various exception types
        message = str(exc)
        if "ir_version" in message.lower():
            LOGGER.warning(
                "onnxsim could not apply shape overrides for %s due to IR version issues: %s. "
                "Falling back to a manual override path.",
                task.source,
                message,
            )
            return apply_overrides_manually()

        raise ShapePreparationError(
            f"onnxsim.simplify failed for {task.source}: {exc}"
        ) from exc

    if not ok:
        LOGGER.warning(
            "onnxsim.simplify reported invalid output for %s; retrying with manual shape overrides.",
            task.source,
        )
        return apply_overrides_manually()

    return safe_shape_inference(simplified_model)


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


def _write_failure_details(
    task: ModelTask,
    run_context: RunContext,
    error: Exception,
    captured_output: str,
) -> Path:
    failure_dir = run_context.directory / "failures"
    failure_dir.mkdir(parents=True, exist_ok=True)

    base_name = task.source.stem or task.source.name.replace(".", "_")
    failure_file = failure_dir / f"{base_name}.log"
    counter = 1
    while failure_file.exists():
        failure_file = failure_dir / f"{base_name}_{counter}.log"
        counter += 1

    sections = [
        f"Export failure for {task.source}",
        "",
        "Traceback:",
        traceback.format_exc(),
    ]
    if captured_output and captured_output.strip():
        sections.extend(["", "Captured stdout/stderr:", captured_output])

    failure_file.write_text("\n".join(sections), encoding="utf-8")
    return failure_file


def _export_model(
    task: ModelTask, config: RunnerConfig, run_context: RunContext
) -> None:
    LOGGER.info("Converting %s", task.source)

    try:
        model_with_shapes = _prepare_model_with_shapes(task, config)
    except ShapePreparationError as error:
        failure_path = _write_failure_details(
            task,
            run_context,
            error,
            captured_output="",
        )
        LOGGER.error(
            "Failed to prepare shapes for %s. Details saved to %s",
            task.source,
            failure_path,
        )
        if run_context.show_full_trace:
            raise
        raise ExportError(
            source=task.source,
            detail_path=failure_path,
            message=str(error),
        ) from None

    with torch.inference_mode():
        module = convert(
            model_with_shapes,
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

        captured = io.StringIO()
        try:
            with (
                contextlib.redirect_stdout(captured),
                contextlib.redirect_stderr(captured),
            ):
                exported = export_program(module, example_args)
        except Exception as error:  # noqa: BLE001 - want to catch and wrap
            failure_path = _write_failure_details(
                task,
                run_context,
                error,
                captured.getvalue(),
            )
            LOGGER.error(
                "Failed to export %s. Full traceback saved to %s",
                task.source,
                failure_path,
            )
            if run_context.show_full_trace:
                raise
            raise ExportError(
                source=task.source,
                detail_path=failure_path,
                message=f"Failed to export {task.source}",
            ) from None

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

    cfg_path = args.cfg.expanduser()
    run_context = _create_run_context(args.verbose)
    _setup_logging(run_context)

    LOGGER.info("Run logs directory: %s", run_context.directory)

    config = _load_config(cfg_path)
    _persist_run_config(cfg_path, run_context)
    tasks = _collect_tasks(config)

    if not tasks:
        LOGGER.warning("No ONNX models found. Nothing to do.")
        return 0

    failures = 0
    for task in tasks:
        try:
            _export_model(task, config, run_context)
        except ExportError:
            failures += 1

    if failures:
        LOGGER.error(
            "Failed to convert %d of %d model(s). See %s for details.",
            failures,
            len(tasks),
            run_context.directory,
        )
        return 1

    LOGGER.info("Converted %d model(s).", len(tasks))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
