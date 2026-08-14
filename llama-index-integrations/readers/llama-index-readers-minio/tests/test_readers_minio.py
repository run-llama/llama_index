from unittest.mock import MagicMock, patch
from pathlib import Path
from llama_index.core.readers.base import BaseReader
from llama_index.readers.minio import BotoMinioReader, MinioReader


def test_class():
    names_of_base_classes = [b.__name__ for b in BotoMinioReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in MinioReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


@patch("minio.Minio")
def test_minio_reader_nested_objects(mock_minio_class) -> None:
    mock_client = MagicMock()
    mock_minio_class.return_value = mock_client

    # Mock objects returned by list_objects
    mock_obj1 = MagicMock()
    mock_obj1.object_name = "contracts/2025/report.txt"
    mock_obj1.is_dir = False

    mock_obj2 = MagicMock()
    mock_obj2.object_name = "contracts/2026/report.txt"
    mock_obj2.is_dir = False

    mock_client.list_objects.return_value = [mock_obj1, mock_obj2]

    # Track downloaded paths
    downloaded_paths = []

    def mock_fget_object(bucket_name, object_name, file_path):
        downloaded_paths.append(file_path)
        # Create dummy file to simulate download
        Path(file_path).write_text(f"content from {object_name}")

    mock_client.fget_object.side_effect = mock_fget_object

    reader = MinioReader(
        bucket="test-bucket",
        minio_endpoint="localhost:9000",
    )

    documents = reader.load_data()

    # Assert both files were downloaded without overwriting each other
    assert len(downloaded_paths) == 2
    assert any("contracts/2025/report.txt" in str(p) for p in downloaded_paths)
    assert any("contracts/2026/report.txt" in str(p) for p in downloaded_paths)
    assert len(documents) == 2
