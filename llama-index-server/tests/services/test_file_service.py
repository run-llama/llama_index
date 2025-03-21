import os
import uuid
from unittest.mock import mock_open, patch

import pytest

from llama_index.server.services.file import FileService, _sanitize_file_name


class TestFileService:
    def test_sanitize_file_name(self):
        # Test with normal alphanumeric name
        assert _sanitize_file_name("test123") == "test123"

        # Test with spaces
        assert _sanitize_file_name("test file") == "test_file"

        # Test with special characters
        assert _sanitize_file_name("test@file!name") == "test_file_name"

        # Test with path-like characters
        assert _sanitize_file_name("test/file/name") == "test_file_name"

        # Test with dots (should be preserved)
        assert _sanitize_file_name("test.file.name") == "test.file.name"

    @patch("uuid.uuid4")
    @patch("os.path.getsize")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_save_file_string_content(
        self, mock_makedirs, mock_file_open, mock_getsize, mock_uuid
    ):
        # Setup
        test_uuid = "12345678-1234-5678-1234-567812345678"
        mock_uuid.return_value = uuid.UUID(test_uuid)
        mock_getsize.return_value = 11  # Length of "Hello World"

        # Execute
        result = FileService.save_file(
            content="Hello World", file_name="test.txt", save_dir="test_dir"
        )

        # Assert
        expected_path = os.path.join("test_dir", f"test_{test_uuid}.txt")
        mock_makedirs.assert_called_once_with(
            os.path.dirname(expected_path), exist_ok=True
        )
        mock_file_open.assert_called_once_with(expected_path, "wb")
        mock_file_open().write.assert_called_once_with(b"Hello World")

        assert result.id == test_uuid
        assert result.name == f"test_{test_uuid}.txt"
        assert result.type == "txt"
        assert result.size == 11
        assert result.path == expected_path
        assert result.url.endswith(expected_path)
        assert result.refs is None

    @patch("uuid.uuid4")
    @patch("os.path.getsize")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_save_file_bytes_content(
        self, mock_makedirs, mock_file_open, mock_getsize, mock_uuid
    ):
        # Setup
        test_uuid = "12345678-1234-5678-1234-567812345678"
        mock_uuid.return_value = uuid.UUID(test_uuid)
        mock_getsize.return_value = 11  # Length of "Hello World"

        # Execute
        result = FileService.save_file(
            content=b"Hello World", file_name="test.txt", save_dir="test_dir"
        )

        # Assert
        expected_path = os.path.join("test_dir", f"test_{test_uuid}.txt")
        mock_file_open().write.assert_called_once_with(b"Hello World")
        assert result.type == "txt"

    @patch("uuid.uuid4")
    @patch("os.path.getsize")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_save_file_with_special_characters(
        self, mock_makedirs, mock_file_open, mock_getsize, mock_uuid
    ):
        # Setup
        test_uuid = "12345678-1234-5678-1234-567812345678"
        mock_uuid.return_value = uuid.UUID(test_uuid)
        mock_getsize.return_value = 11

        # Execute
        result = FileService.save_file(
            content="Hello World", file_name="test@file!.txt", save_dir="test_dir"
        )

        # Assert
        expected_path = os.path.join("test_dir", f"test_file__{test_uuid}.txt")
        assert result.name == f"test_file__{test_uuid}.txt"

    @patch("uuid.uuid4")
    @patch("os.path.getsize")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_save_file_default_directory(
        self, mock_makedirs, mock_file_open, mock_getsize, mock_uuid
    ):
        # Setup
        test_uuid = "12345678-1234-5678-1234-567812345678"
        mock_uuid.return_value = uuid.UUID(test_uuid)
        mock_getsize.return_value = 11

        # Execute
        result = FileService.save_file(content="Hello World", file_name="test.txt")

        # Assert
        expected_path = os.path.join("output", "uploaded", f"test_{test_uuid}.txt")
        mock_makedirs.assert_called_once_with(
            os.path.dirname(expected_path), exist_ok=True
        )
        assert result.path == expected_path

    @patch("uuid.uuid4")
    @patch("os.getenv")
    @patch("os.path.getsize")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_save_file_custom_url_prefix(
        self, mock_makedirs, mock_file_open, mock_getsize, mock_getenv, mock_uuid
    ):
        # Setup
        test_uuid = "12345678-1234-5678-1234-567812345678"
        mock_uuid.return_value = uuid.UUID(test_uuid)
        mock_getsize.return_value = 11
        mock_getenv.return_value = "https://custom-url.com/files"

        # Execute
        result = FileService.save_file(
            content="Hello World", file_name="test.txt", save_dir="test_dir"
        )

        # Assert
        expected_path = os.path.join("test_dir", f"test_{test_uuid}.txt")
        expected_url = os.path.join(
            "https://custom-url.com/files", "test_dir", f"test_{test_uuid}.txt"
        )
        assert result.url == expected_url

    def test_save_file_no_extension(self):
        # Test that saving a file without extension raises ValueError
        with pytest.raises(ValueError, match="File is not supported!"):
            FileService.save_file(
                content="Hello World", file_name="test", save_dir="test_dir"
            )

    @patch("uuid.uuid4")
    @patch("os.path.getsize")
    @patch("builtins.open")
    @patch("os.makedirs")
    def test_save_file_permission_error(
        self, mock_makedirs, mock_file_open, mock_getsize, mock_uuid
    ):
        # Setup
        test_uuid = "12345678-1234-5678-1234-567812345678"
        mock_uuid.return_value = uuid.UUID(test_uuid)
        mock_file_open.side_effect = PermissionError("Permission denied")

        # Execute and Assert
        with pytest.raises(PermissionError):
            FileService.save_file(
                content="Hello World", file_name="test.txt", save_dir="test_dir"
            )

    @patch("uuid.uuid4")
    @patch("os.path.getsize")
    @patch("builtins.open")
    @patch("os.makedirs")
    def test_save_file_io_error(
        self, mock_makedirs, mock_file_open, mock_getsize, mock_uuid
    ):
        # Setup
        test_uuid = "12345678-1234-5678-1234-567812345678"
        mock_uuid.return_value = uuid.UUID(test_uuid)
        mock_file_open.side_effect = OSError("IO Error")

        # Execute and Assert
        with pytest.raises(IOError):
            FileService.save_file(
                content="Hello World", file_name="test.txt", save_dir="test_dir"
            )
