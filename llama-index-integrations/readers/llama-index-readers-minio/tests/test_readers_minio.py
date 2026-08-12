from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from llama_index.core.readers.base import BaseReader
from llama_index.readers.minio import BotoMinioReader, MinioReader


def test_class():
    names_of_base_classes = [b.__name__ for b in BotoMinioReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in MinioReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def _mock_minio_client(contents):
    client = mock.MagicMock()
    client.list_objects.return_value = [
        SimpleNamespace(object_name=name) for name in contents
    ]

    def fget_object(bucket_name, object_name, file_path):
        Path(file_path).write_text(contents[object_name])

    client.fget_object.side_effect = fget_object
    return client


def test_objects_with_same_basename_are_all_loaded():
    contents = {
        "contracts/2025/report.txt": "2025 report",
        "contracts/2026/report.txt": "2026 report",
    }
    client = _mock_minio_client(contents)

    with mock.patch("minio.Minio", return_value=client):
        documents = MinioReader(
            bucket="documents", minio_endpoint="localhost:9000"
        ).load_data()

    assert client.fget_object.call_count == 2
    destinations = {call.args[2] for call in client.fget_object.call_args_list}
    assert len(destinations) == 2
    assert len(documents) == 2
    assert {doc.text for doc in documents} == {"2025 report", "2026 report"}
    assert all(doc.metadata["file_name"] == "report.txt" for doc in documents)
    assert len({doc.metadata["file_path"] for doc in documents}) == 2


def test_same_basename_ids_unique_with_filename_as_id():
    contents = {
        "contracts/2025/report.txt": "2025 report",
        "contracts/2026/report.txt": "2026 report",
    }
    client = _mock_minio_client(contents)

    with mock.patch("minio.Minio", return_value=client):
        documents = MinioReader(
            bucket="documents",
            minio_endpoint="localhost:9000",
            filename_as_id=True,
        ).load_data()

    assert len({doc.id_ for doc in documents}) == 2


def test_object_name_with_backslash_stays_in_temp_dir(tmp_path):
    contents = {"..\\evil.txt": "payload"}
    client = _mock_minio_client(contents)

    with mock.patch(
        "llama_index.readers.minio.minio_client.base.tempfile.TemporaryDirectory"
    ) as mock_temp_dir:
        mock_temp_dir.return_value.__enter__.return_value = str(tmp_path)
        mock_temp_dir.return_value.__exit__.return_value = False
        with mock.patch("minio.Minio", return_value=client):
            documents = MinioReader(
                bucket="documents", minio_endpoint="localhost:9000"
            ).load_data()

    destination = Path(client.fget_object.call_args.args[2])
    assert destination == tmp_path.resolve() / "00000000" / "evil.txt"
    assert len(documents) == 1
    assert documents[0].text == "payload"


def test_unsafe_object_name_raises():
    contents = {"contracts/..": "payload"}
    client = _mock_minio_client(contents)

    with mock.patch("minio.Minio", return_value=client):
        with pytest.raises(ValueError, match="Unsafe object name"):
            MinioReader(bucket="documents", minio_endpoint="localhost:9000").load_data()

    client.fget_object.assert_not_called()


def test_file_metadata_receives_download_paths():
    contents = {
        "contracts/2025/report.txt": "2025 report",
        "contracts/2026/report.txt": "2026 report",
    }
    client = _mock_minio_client(contents)

    with mock.patch("minio.Minio", return_value=client):
        documents = MinioReader(
            bucket="documents",
            minio_endpoint="localhost:9000",
            file_metadata=lambda path: {"src_path": path},
        ).load_data()

    assert len(documents) == 2
    metadata_paths = {Path(doc.metadata["src_path"]).resolve() for doc in documents}
    download_paths = {
        Path(call.args[2]).resolve() for call in client.fget_object.call_args_list
    }
    assert metadata_paths == download_paths


def test_single_key_branch_unchanged():
    contents = {"contracts/2025/report.txt": "single file"}
    client = _mock_minio_client(contents)

    with mock.patch("minio.Minio", return_value=client):
        documents = MinioReader(
            bucket="documents",
            key="contracts/2025/report.txt",
            minio_endpoint="localhost:9000",
        ).load_data()

    client.list_objects.assert_not_called()
    assert len(documents) == 1
    assert documents[0].text == "single file"
