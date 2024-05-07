import json
from tempfile import TemporaryDirectory

import unittest
from unittest.mock import MagicMock
from llama_index.readers.google import GoogleDriveReader

from llama_index.core.readers.base import BaseReader
import pytest
from llama_index.readers.google import GoogleDriveReader

test_client_config = {"client_config": {"key": "value"}}
test_authorized_user_info = {"authorized_user_info": {"key": "value"}}
test_service_account_key = {"service_account_key": {"key": "value"}}


class TestGoogleDriveReader(unittest.TestCase):
    def test_class(self):
        names_of_base_classes = [b.__name__ for b in GoogleDriveReader.__mro__]
        assert BaseReader.__name__ in names_of_base_classes

    def test_serialize(self):
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

    def test_serialize_from_file(self):
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

    def test_error_on_missing_args(self):
        with pytest.raises(ValueError):
            GoogleDriveReader()

    def test_load_data_with_drive_id(self):
        # Mock the necessary objects and methods
        mock_credentials = MagicMock()
        mock_drive = MagicMock()
        GoogleDriveReader._get_credentials = MagicMock(
            return_value=(mock_credentials, mock_drive)
        )
        GoogleDriveReader._load_from_folder = MagicMock(
            return_value=["document1", "document2"]
        )
        reader = GoogleDriveReader(
            client_config={
                "client_id": "example_client_id",
                "client_secret": "example_client_secret",
            },
        )

        # Test with a specific drive_id
        drive_id = "example_drive_id"
        folder_id = "example_folder_id"
        result = reader.load_data(drive_id=drive_id, folder_id=folder_id)

        # Assert that the correct methods are called and the correct result is returned
        reader._get_credentials.assert_called_once()
        reader._load_from_folder.assert_called_once_with(
            drive_id, folder_id, None, None
        )

        assert result == ["document1", "document2"]
