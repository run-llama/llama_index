import hashlib
from pathlib import Path

from llama_index.core.readers.base import BaseReader
from llama_index.readers.minio import BotoMinioReader, MinioReader


def test_class():
    names_of_base_classes = [b.__name__ for b in BotoMinioReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in MinioReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_minio_reader_preserves_same_named_objects(monkeypatch):
    object_contents = {
        "contracts/2025/report.txt": "2025 report",
        "contracts/2026/report.txt": "2026 report",
    }
    downloaded_paths = []

    class FakeObject:
        def __init__(self, object_name):
            self.object_name = object_name

    class FakeMinio:
        def __init__(self, *args, **kwargs):
            pass

        def list_objects(self, bucket_name, prefix, recursive):
            return [FakeObject(object_name) for object_name in object_contents]

        def fget_object(self, bucket_name, object_name, file_path):
            downloaded_paths.append(Path(file_path))
            Path(file_path).write_text(object_contents[object_name])

    monkeypatch.setattr("minio.Minio", FakeMinio)

    documents = MinioReader(
        bucket="documents",
        minio_endpoint="example.invalid",
    ).load_data()

    assert len(documents) == 2
    assert {document.text for document in documents} == set(object_contents.values())
    assert {path.name for path in downloaded_paths} == {
        f"{hashlib.sha256(object_name.encode('utf-8')).hexdigest()}.txt"
        for object_name in object_contents
    }
