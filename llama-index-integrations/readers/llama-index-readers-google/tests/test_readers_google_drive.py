import json
from tempfile import TemporaryDirectory

from llama_index.core.readers.base import BaseReader
import pytest
from llama_index.readers.google import GoogleDriveReader

test_client_config = {"client_config": {"key": "value"}}
test_authorized_user_info = {"authorized_user_info": {"key": "value"}}
test_service_account_key = {"service_account_key": {"key": "value"}}


def test_class():
    names_of_base_classes = [b.__name__ for b in GoogleDriveReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_serialize():
    reader = GoogleDriveReader(
        client_config=test_client_config,
        authorized_user_info=test_authorized_user_info,
    )

    schema = reader.schema()
    assert schema is not None
    assert len(schema) > 0
    assert "client_config" in schema["properties"]

    json_reader = reader.json(exclude_unset=True)

    new_reader = GoogleDriveReader.parse_raw(json_reader)
    assert new_reader.client_config == reader.client_config
    assert new_reader.authorized_user_info == reader.authorized_user_info


def test_serialize_from_file():
    with TemporaryDirectory() as tmp_dir:
        file_name = f"{tmp_dir}/test_service_account_key.json"
        with open(file_name, "w") as f:
            f.write(json.dumps(test_service_account_key))

        reader = GoogleDriveReader(
            service_account_key_path=file_name,
        )

    schema = reader.schema()
    assert schema is not None
    assert len(schema) > 0
    assert "service_account_key" in schema["properties"]

    json_reader = reader.json(exclude_unset=True)

    new_reader = GoogleDriveReader.parse_raw(json_reader)
    assert new_reader.service_account_key == reader.service_account_key
    assert new_reader.service_account_key == test_service_account_key


def test_error_on_missing_args():
    with pytest.raises(ValueError):
        GoogleDriveReader()
