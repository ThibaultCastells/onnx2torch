#!/usr/bin/env python3
"""CLI wrapper to convert ONNX models to torch.export programs."""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import math
import shutil
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Sequence, Tuple

import torch
import torch.fx as fx
import yaml
import onnx
from onnx.onnx_ml_pb2 import ModelProto, ValueInfoProto

from onnx2torch.converter import convert
from onnx2torch.utils.dtype import onnx_dtype_to_torch_dtype
from onnx2torch.utils.shape_warmup import shape_warmup

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback if dependency missing
    tqdm = None

try:  # torch < 2.1 does not expose torch.export.export
    from torch.export import export as export_program
except (
    ImportError,
    AttributeError,
) as exc:  # pragma: no cover - guard for older torch versions
    raise RuntimeError(
        "torch>=2.1 is required to export models to .pt2 format"
    ) from exc

try:
    from torch.export import save as export_save  # torch>=2.1 preferred path
except (ImportError, AttributeError):  # pragma: no cover - guard for older torch
    export_save = None

LOGGER = logging.getLogger("onnx2torch.runner")
DEFAULT_OUTPUT_DIR = Path("data/executorch")
RUN_LOG_ROOT = Path("logs/run_logs")
SUPPORTED_FILLS = {"zeros", "ones", "random"}


class BoolTypedStorageError(RuntimeError):
    """Raised when legacy serialization encounters boolean typed storage."""


def _resolve_save_callable(exported: torch.export.ExportedProgram):
    if export_save is not None:
        return lambda program, path: export_save(program, str(path))
    if hasattr(exported, "save"):
        return lambda program, path: program.save(str(path))  # type: ignore[attr-defined]
    raise RuntimeError(
        "This version of torch does not support saving ExportedProgram instances. "
        "Please upgrade torch to >=2.1."
    )


@dataclass
class InputConfig:
    path: Path
    pattern: str = "*.onnx"
    recursive: bool = True


@dataclass
class ExampleInputConfig:
    default_fill: str = "zeros"
    overrides: Dict[str, Dict[str, object]] = field(default_factory=dict)
    scales: Dict[str, int] = field(default_factory=dict)
    max_total_elements: int = 2_000_000
    warmup_max_total_elements: int = 131_072
    enable_warmup: bool = True


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


def _save_exported_program(
    exported: torch.export.ExportedProgram,
    destination: Path,
    *,
    task: "ModelTask",
    run_context: "RunContext",
) -> None:
    """Persist an ExportedProgram and verify the resulting artifact."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with _serialize_bool_storage_as_untyped():
        try:
            _save_without_typed_storage(exported, destination)
        except (
            ImportError,
            AttributeError,
        ) as error:  # pragma: no cover - torch internals missing
            LOGGER.warning(
                "Falling back to legacy torch.export saving path without typed-storage shim; "
                "boolean initializers may fail to load on older runtimes."
            )
            save_callable = _resolve_save_callable(exported)
            save_callable(exported, destination)
            LOGGER.debug("Saved %s using fallback path after %s", destination, error)

    _verify_export_artifact(destination, task, run_context)


def _save_without_typed_storage(
    exported: torch.export.ExportedProgram, destination: Path
) -> None:
    """Save ExportedProgram forcing legacy tensor storage format."""

    from torch._export.serde import serialize as serde_module

    try:
        FakeTensor = serde_module.FakeTensor
        _reduce_fake_tensor = serde_module._reduce_fake_tensor
        DEFAULT_PICKLE_PROTOCOL = serde_module.DEFAULT_PICKLE_PROTOCOL
    except AttributeError as exc:  # pragma: no cover - internal API change
        raise AttributeError("serialize module missing FakeTensor helpers") from exc

    import copyreg
    import io
    import pickle

    from torch.serialization import _legacy_save

    original_serializer = serde_module.serialize_torch_artifact

    def _legacy_serialize(
        artifact, pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL
    ) -> bytes:
        if artifact is None:
            return b""

        assert (  # pragma: no cover - defensive check
            FakeTensor not in copyreg.dispatch_table
        ), "Refusing to stomp on existing FakeTensor reducer"

        try:
            copyreg.pickle(FakeTensor, _reduce_fake_tensor)
            buffer = io.BytesIO()
            _legacy_save(artifact, buffer, pickle, pickle_protocol)
            return buffer.getvalue()
        finally:
            del copyreg.dispatch_table[FakeTensor]

    save_callable = _resolve_save_callable(exported)

    try:
        serde_module.serialize_torch_artifact = _legacy_serialize  # type: ignore[assignment]
        save_callable(exported, destination)
    finally:
        serde_module.serialize_torch_artifact = original_serializer  # type: ignore[assignment]


@contextmanager
def _serialize_bool_storage_as_untyped() -> Iterator[None]:
    typed_storage = getattr(torch.storage, "TypedStorage", None)
    if typed_storage is None:
        yield
        return

    original_reduce_ex = getattr(typed_storage, "__reduce_ex__", None)
    original_reduce = getattr(typed_storage, "__reduce__", None)
    original_pickle_storage_type = getattr(typed_storage, "pickle_storage_type", None)
    original_private_pickle_storage_type = getattr(
        typed_storage, "_pickle_storage_type", None
    )

    if (
        original_reduce_ex is None
        or original_reduce is None
        or original_pickle_storage_type is None
        or original_private_pickle_storage_type is None
    ):
        yield
        return

    def _reduce_ex(self, protocol):  # type: ignore[override]
        dtype = getattr(self, "dtype", None)
        if dtype is torch.bool:
            return self.untyped().__reduce_ex__(protocol)  # type: ignore[attr-defined]
        return original_reduce_ex(self, protocol)

    def _reduce(self):  # type: ignore[override]
        dtype = getattr(self, "dtype", None)
        if dtype is torch.bool:
            return self.untyped().__reduce__()  # type: ignore[attr-defined]
        return original_reduce(self)

    typed_storage.__reduce_ex__ = _reduce_ex  # type: ignore[attr-defined]
    typed_storage.__reduce__ = _reduce  # type: ignore[attr-defined]
    typed_storage.pickle_storage_type = (  # type: ignore[attr-defined]
        lambda self: "UntypedStorage"
        if getattr(self, "dtype", None) is torch.bool
        else original_pickle_storage_type(self)
    )
    typed_storage._pickle_storage_type = (  # type: ignore[attr-defined]
        lambda self: "UntypedStorage"
        if getattr(self, "dtype", None) is torch.bool
        else original_private_pickle_storage_type(self)
    )
    try:
        yield
    finally:
        typed_storage.__reduce_ex__ = original_reduce_ex  # type: ignore[attr-defined]
        typed_storage.__reduce__ = original_reduce  # type: ignore[attr-defined]
        typed_storage.pickle_storage_type = original_pickle_storage_type  # type: ignore[attr-defined]
        typed_storage._pickle_storage_type = (  # type: ignore[attr-defined]
            original_private_pickle_storage_type
        )


@contextmanager
def _suppress_export_deserialize_warnings():
    logger = logging.getLogger("torch._export.serde.serialize")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous)


@contextmanager
def _fail_on_bool_typed_storage() -> Iterator[None]:
    original_new = torch.storage.TypedStorage.__new__

    BOOL_METADATA_NAMES = {"boolean", "bool", "torch.bool"}
    BOOL_SCALAR_CODES = {11}  # c10::ScalarType::Bool stable tag

    def _looks_like_bool_metadata(candidate: object) -> bool:
        if isinstance(candidate, dict):
            scalar_type = candidate.get("_scalar_type")
            if isinstance(scalar_type, int) and scalar_type in BOOL_SCALAR_CODES:
                return True
            if isinstance(scalar_type, str) and "bool" in scalar_type.lower():
                return True

            name = candidate.get("_name")
            if isinstance(name, str) and name.lower() in BOOL_METADATA_NAMES:
                return True
        return False

    def _iter_dtype_candidates(*values: object) -> Iterator[torch.dtype]:
        for value in values:
            if isinstance(value, torch.dtype):
                yield value

    def _guard(cls, *args, **kwargs):  # type: ignore[override]
        dtype = kwargs.get("dtype")
        if not isinstance(dtype, torch.dtype):
            dtype = next(_iter_dtype_candidates(*args), None)

        if dtype is torch.bool:
            raise BoolTypedStorageError(
                "Boolean TypedStorage encountered during load."  # pragma: no cover - guard path
            )

        if dtype is None:
            metadata_values = list(args) + list(kwargs.values())
            if any(_looks_like_bool_metadata(value) for value in metadata_values):
                raise BoolTypedStorageError(
                    "Boolean TypedStorage encountered during load."  # pragma: no cover - guard path
                )

        return original_new(cls, *args, **kwargs)

    torch.storage.TypedStorage.__new__ = _guard  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.storage.TypedStorage.__new__ = original_new  # type: ignore[assignment]


def _verify_export_artifact(
    destination: Path, task: "ModelTask", run_context: "RunContext"
) -> None:
    try:
        from torch.export import load as export_load
    except (ImportError, AttributeError):  # pragma: no cover - very old torch
        LOGGER.debug("torch.export.load unavailable; skipping artifact verification")
        return

    try:
        with _suppress_export_deserialize_warnings():
            with _fail_on_bool_typed_storage():
                export_load(str(destination))
    except BoolTypedStorageError as error:
        failure_path = _write_failure_details(
            task,
            run_context,
            error,
            captured_output="",
        )
        LOGGER.error(
            "Saved artifact %s contains boolean typed storage. Details saved to %s",
            destination,
            failure_path,
        )
        raise ExportError(
            source=task.source,
            detail_path=failure_path,
            message=(
                f"Exported artifact at {destination} stores boolean tensors using an "
                "unsupported TypedStorage encoding."
            ),
        ) from None
    except Exception as error:  # noqa: BLE001 - propagate verification failure details
        failure_path = _write_failure_details(
            task,
            run_context,
            error,
            captured_output="",
        )
        LOGGER.error(
            "Failed to reload %s for post-export validation. See %s",
            destination,
            failure_path,
        )
        raise ExportError(
            source=task.source,
            detail_path=failure_path,
            message=f"Failed to reload exported artifact {destination}",
        ) from None


def _run_export_verifier(
    exported: torch.export.ExportedProgram,
    task: "ModelTask",
    run_context: "RunContext",
) -> None:
    try:
        from torch._export.verifier import SpecViolationError, load_verifier
    except ImportError:  # pragma: no cover - verifier removed or unavailable
        LOGGER.debug("torch._export.verifier unavailable; skipping verification")
        return

    try:
        verifier_cls = load_verifier("ATEN")
    except KeyError:  # pragma: no cover - dialect missing in this torch build
        LOGGER.debug("ATEN verifier not registered; skipping verification")
        return

    verifier = verifier_cls()

    try:
        verifier.check(exported)
    except SpecViolationError as error:
        failure_path = _write_failure_details(
            task,
            run_context,
            error,
            captured_output="",
        )
        LOGGER.warning(
            "ATen verifier reported issues for %s. Details saved to %s",
            task.source,
            failure_path,
        )
        return

    LOGGER.debug("ATen verifier passed for %s", task.source)


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
        example_cfg_dict = example_cfg_raw
        default_fill = example_cfg_dict.get("default_fill", "zeros")
        overrides = example_cfg_dict.get(
            "overrides", example_cfg_dict.get("inputs", {})
        )
    elif example_cfg_raw in (None, ""):
        example_cfg_dict = {}
        default_fill = "zeros"
        overrides = {}
    else:
        raise TypeError('The "example_inputs" section must be a mapping if provided.')

    overrides_final: Dict[str, Dict[str, object]] = {}
    for key, value in (overrides or {}).items():
        if value is None:
            overrides_final[str(key)] = {}
        elif isinstance(value, dict):
            overrides_final[str(key)] = dict(value)
        else:
            raise TypeError(
                f"Override for {key!r} must be a mapping, got {type(value).__name__}."
            )

    scales_raw = example_cfg_dict.get("scales", {})
    if not isinstance(scales_raw, dict):
        raise TypeError(
            'The "scales" entry under "example_inputs" must be a mapping if provided.'
        )

    example_cfg = ExampleInputConfig(
        default_fill=str(default_fill).lower(),
        overrides=overrides_final,
        scales={str(k): int(v) for k, v in scales_raw.items()},
        max_total_elements=int(
            example_cfg_dict.get(
                "max_total_elements", ExampleInputConfig.max_total_elements
            )
        ),
        warmup_max_total_elements=int(
            example_cfg_dict.get(
                "warmup_max_total_elements",
                ExampleInputConfig.warmup_max_total_elements,
            )
        ),
        enable_warmup=bool(example_cfg_dict.get("enable_warmup", True)),
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
            dims.append(int(dim.dim_value))
        else:
            dims.append(1)
    return dims


def _apply_dimension_scales(
    shape: List[int],
    labels: Sequence[str] | None,
    scales: Mapping[str, int],
    input_name: str,
) -> List[int]:
    if not labels:
        return shape

    adjusted = list(shape)
    for index, label in enumerate(labels):
        if index >= len(adjusted) or label is None:
            continue
        key = str(label)
        if key not in scales:
            continue
        clamp_value = int(scales[key])
        if clamp_value <= 0:
            continue
        if adjusted[index] > clamp_value:
            LOGGER.warning(
                "Clamping dimension %s[%s] from %d to %d based on scale hints.",
                input_name,
                key,
                adjusted[index],
                clamp_value,
            )
            adjusted[index] = clamp_value
    return adjusted


def _enforce_element_cap(
    shape: List[int],
    cap: int,
    *,
    input_name: str,
    reason: str,
) -> List[int]:
    if cap <= 0 or not shape:
        return shape

    original = list(shape)
    total_requested = math.prod(original) or 1
    if total_requested <= cap:
        return shape

    new_shape = list(original)
    changed = False

    for index in reversed(range(len(new_shape))):
        current_total = math.prod(new_shape) or 1
        if current_total <= cap:
            break
        other_prod = math.prod(new_shape[:index] + new_shape[index + 1 :]) or 1
        max_dim = max(1, cap // other_prod)
        if max_dim < new_shape[index]:
            new_shape[index] = max_dim
            changed = True

    current_total = math.prod(new_shape) or 1
    if current_total > cap:
        new_shape = [1] * len(new_shape)
        current_total = 1
        changed = True

    if changed:
        LOGGER.warning(
            "Input %s requested %d elements; clamped to %d elements (%s cap=%d).",
            input_name,
            total_requested,
            current_total,
            reason,
            cap,
        )
        LOGGER.debug("Adjusted %s shape from %s to %s", input_name, original, new_shape)

    return new_shape


def _parse_scale_overrides(entries: Sequence[str]) -> Dict[str, int]:
    overrides: Dict[str, int] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"Scale override {entry!r} must be in the form <name>=<value>."
            )
        key, value = entry.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Scale override {entry!r} is missing a name.")
        try:
            overrides[key] = int(value)
        except ValueError as exc:  # noqa: PERF203 - provide clearer message
            raise ValueError(
                f"Scale override {entry!r} must have an integer value."
            ) from exc
    return overrides


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


def _materialise_example_tensor(
    shape: Sequence[int],
    *,
    fill: str,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    shape_tuple: Tuple[int, ...] = tuple(max(1, int(dim)) for dim in shape)
    if fill == "zeros":
        return torch.zeros(shape_tuple, dtype=dtype, device=device)
    if fill == "ones":
        return torch.ones(shape_tuple, dtype=dtype, device=device)
    if fill == "random":
        if dtype.is_floating_point or dtype.is_complex:
            return torch.randn(shape_tuple, dtype=dtype, device=device)
        if dtype == torch.bool:
            return torch.randint(0, 2, shape_tuple, dtype=torch.bool, device=device)
        return torch.randint(0, 10, shape_tuple, dtype=dtype, device=device)

    raise ValueError(f"Unsupported fill strategy {fill!r}.")


def _create_example_tensors(
    name: str,
    value_info: ValueInfoProto,
    override: Dict[str, object],
    example_cfg: ExampleInputConfig,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if "shape" in override:
        runtime_shape = [int(size) for size in override["shape"]]
    else:
        runtime_shape = _dim_sizes(value_info)
        if override.get("ensure_min_dim"):
            min_dim = int(override["ensure_min_dim"])
            runtime_shape = [max(dim, min_dim) for dim in runtime_shape]

    warmup_shape_override = override.get("warmup_shape")
    if warmup_shape_override is not None:
        warmup_shape = [int(size) for size in warmup_shape_override]
    else:
        warmup_shape = list(runtime_shape)

    labels_raw = override.get("dim_labels")
    labels: List[str | None] | None = None
    if labels_raw is not None:
        labels = []
        for entry in labels_raw:
            labels.append(None if entry is None else str(entry))

    runtime_shape = [max(1, int(dim)) for dim in runtime_shape]
    warmup_shape = [max(1, int(dim)) for dim in warmup_shape]

    runtime_shape = _apply_dimension_scales(
        runtime_shape, labels, example_cfg.scales, name
    )
    warmup_shape = _apply_dimension_scales(
        warmup_shape, labels, example_cfg.scales, f"{name} (warmup)"
    )

    runtime_cap = int(
        override.get("max_total_elements", example_cfg.max_total_elements)
    )
    warmup_cap = int(
        override.get("warmup_max_total_elements", example_cfg.warmup_max_total_elements)
    )

    runtime_shape = _enforce_element_cap(
        runtime_shape, runtime_cap, input_name=name, reason="runtime"
    )
    warmup_shape = _enforce_element_cap(
        warmup_shape, warmup_cap, input_name=name, reason="warmup"
    )

    torch_dtype = _resolve_dtype(name, value_info.type.tensor_type.elem_type, override)

    fill = str(override.get("fill", example_cfg.default_fill)).lower()
    if fill not in SUPPORTED_FILLS:
        raise ValueError(f"Unsupported fill strategy {fill!r} for input {name}.")

    runtime_tensor = _materialise_example_tensor(
        runtime_shape, fill=fill, dtype=torch_dtype, device=device
    )
    warmup_tensor = _materialise_example_tensor(
        warmup_shape, fill=fill, dtype=torch_dtype, device=device
    )

    return runtime_tensor, warmup_tensor


def _build_example_and_warmup_args(
    model: ModelProto, config: RunnerConfig
) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    overrides = config.example_inputs.overrides
    runtime_tensors: List[torch.Tensor] = []
    warmup_tensors: List[torch.Tensor] = []

    for value_info in model.graph.input:
        if value_info.name in initializer_names:
            continue

        override = overrides.get(value_info.name, {})
        runtime_tensor, warmup_tensor = _create_example_tensors(
            name=value_info.name,
            value_info=value_info,
            override=override,
            example_cfg=config.example_inputs,
            device=config.device,
        )
        runtime_tensors.append(runtime_tensor)
        warmup_tensors.append(warmup_tensor)

    if not config.example_inputs.enable_warmup:
        return tuple(runtime_tensors), tuple()

    return tuple(runtime_tensors), tuple(warmup_tensors)


def _build_static_dynamic_shapes(
    args: Tuple[torch.Tensor, ...],
) -> Tuple[Dict[int, int] | None, ...]:
    specs: List[Dict[int, int] | None] = []
    for tensor in args:
        if isinstance(tensor, torch.Tensor):
            dims = {index: int(size) for index, size in enumerate(tensor.shape)}
            specs.append(dims)
        else:
            specs.append(None)
    return tuple(specs)


def _strip_scalar_runtime_asserts(exported: torch.export.ExportedProgram) -> None:
    graph = exported.graph_module.graph
    assert_nodes = []
    bool_producers = set()

    for node in list(graph.nodes):
        if (
            node.op == "call_function"
            and node.target is torch.ops.aten._assert_scalar.default
        ):
            assert_nodes.append(node)
            for arg in node.args:
                if isinstance(arg, fx.Node):
                    bool_producers.add(arg)

    for node in assert_nodes:
        graph.erase_node(node)

    for node in list(bool_producers):
        if not node.users:
            graph.erase_node(node)

    graph.eliminate_dead_code()
    graph.lint()
    exported.graph_module.recompile()


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
        model = onnx.load(str(task.source))
    except Exception as error:  # noqa: BLE001 - propagate context to user
        failure_path = _write_failure_details(
            task,
            run_context,
            error,
            captured_output="",
        )
        LOGGER.error(
            "Failed to load %s. Details saved to %s",
            task.source,
            failure_path,
        )
        raise ExportError(
            source=task.source,
            detail_path=failure_path,
            message=f"Failed to load {task.source}",
        ) from None

    with torch.inference_mode():
        try:
            progress_label = f"Nodes {task.source.name}" if tqdm is not None else False
            module = convert(
                model,
                save_input_names=config.save_input_names,
                attach_onnx_mapping=config.attach_onnx_mapping,
                progress=progress_label,
            )
        except Exception as error:  # noqa: BLE001 - capture conversion failures
            failure_path = _write_failure_details(
                task,
                run_context,
                error,
                captured_output="",
            )
            LOGGER.error(
                "Failed to convert %s. Details saved to %s",
                task.source,
                failure_path,
            )
            raise ExportError(
                source=task.source,
                detail_path=failure_path,
                message=f"Failed to convert {task.source}",
            ) from None

        module.eval()
        module.to(config.device)

        runtime_args, warmup_args = _build_example_and_warmup_args(model, config)

        LOGGER.debug(
            "Runtime args for %s: %s",
            task.source,
            [tuple(t.shape) for t in runtime_args],
        )
        LOGGER.debug(
            "Warmup args for %s: %s",
            task.source,
            [tuple(t.shape) for t in warmup_args],
        )

        if warmup_args:
            LOGGER.debug("Starting warmup execution for %s", task.source)
            try:
                with shape_warmup():
                    module(*warmup_args)
            except Exception as error:  # noqa: BLE001 - continue to wrap with details
                failure_path = _write_failure_details(
                    task,
                    run_context,
                    error,
                    captured_output="",
                )
                LOGGER.error(
                    "Warm-up execution failed for %s. Details saved to %s",
                    task.source,
                    failure_path,
                )
                raise ExportError(
                    source=task.source,
                    detail_path=failure_path,
                    message=f"Warm-up execution failed for {task.source}",
                ) from None
            else:
                LOGGER.debug("Finished warmup execution for %s", task.source)

        captured = io.StringIO()
        try:
            LOGGER.debug("Starting export_program for %s", task.source)
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(
                captured
            ):
                dynamic_shapes = _build_static_dynamic_shapes(runtime_args)
                exported = export_program(
                    module,
                    runtime_args,
                    dynamic_shapes=dynamic_shapes,
                )
                _strip_scalar_runtime_asserts(exported)
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
            raise ExportError(
                source=task.source,
                detail_path=failure_path,
                message=f"Failed to export {task.source}",
            ) from None
        else:
            LOGGER.debug("Finished export_program for %s", task.source)

    _run_export_verifier(exported, task, run_context)

    _save_exported_program(
        exported,
        task.destination,
        task=task,
        run_context=run_context,
    )
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
    parser.add_argument(
        "--scale",
        action="append",
        default=[],
        help="Override dimension scale hints (name=value). Can be repeated.",
    )

    args = parser.parse_args(argv)

    cfg_path = args.cfg.expanduser()
    run_context = _create_run_context(args.verbose)
    _setup_logging(run_context)

    LOGGER.info("Run logs directory: %s", run_context.directory)

    config = _load_config(cfg_path)
    if args.scale:
        scale_overrides = _parse_scale_overrides(args.scale)
        if scale_overrides:
            config.example_inputs.scales.update(scale_overrides)
            LOGGER.info("Applied scale overrides: %s", scale_overrides)
    _persist_run_config(cfg_path, run_context)
    tasks = _collect_tasks(config)

    if not tasks:
        LOGGER.warning("No ONNX models found. Nothing to do.")
        return 0

    failures = 0
    progress_bar = None
    if tqdm is not None and len(tasks) > 1:
        progress_bar = tqdm(
            total=len(tasks),
            desc="Models",
            unit="model",
            leave=False,
        )

    try:
        for task in tasks:
            try:
                _export_model(task, config, run_context)
            except ExportError:
                failures += 1
            finally:
                if progress_bar is not None:
                    progress_bar.update()
    finally:
        if progress_bar is not None:
            progress_bar.close()

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
