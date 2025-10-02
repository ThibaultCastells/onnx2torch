import io
import pickle

import torch

import pytest

from torch.serialization import _legacy_save

from run import BoolTypedStorageError, _fail_on_bool_typed_storage
from run import _serialize_bool_storage_as_untyped


def _dump_tensor(tensor: torch.Tensor) -> io.BytesIO:
    buffer = io.BytesIO()
    _legacy_save(tensor, buffer, pickle_module=pickle, pickle_protocol=2)
    buffer.seek(0)
    return buffer


def test_fail_on_bool_typed_storage_detects_boolean_storage():
    buffer = _dump_tensor(torch.tensor([True, False]))

    with pytest.raises(BoolTypedStorageError):
        with _fail_on_bool_typed_storage():
            torch.load(buffer)


def test_fail_on_bool_typed_storage_detects_metadata_only_bool():
    metadata = {"_scalar_type": 11, "_name": "boolean", "_itemsize": 1}

    with pytest.raises(BoolTypedStorageError):
        with _fail_on_bool_typed_storage():
            torch.storage.TypedStorage.__new__(
                torch.storage.TypedStorage,
                None,
                metadata,
            )


def test_fail_on_bool_typed_storage_allows_non_boolean():
    buffer = _dump_tensor(torch.tensor([1.0, 2.0], dtype=torch.float32))

    with _fail_on_bool_typed_storage():
        tensor = torch.load(buffer)

    assert tensor.dtype is torch.float32


def test_serialize_bool_storage_as_untyped(tmp_path):
    tensor = torch.tensor([True, False], dtype=torch.bool)
    buffer = tmp_path / "payload.pt"

    with _serialize_bool_storage_as_untyped():
        torch.save(tensor, buffer)

    payload = buffer.read_bytes()
    assert b"BoolStorage" not in payload
