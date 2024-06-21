from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document
from llama_index.readers.s3 import S3Reader
from typing import List
from moto.server import ThreadedMotoServer
import pytest
import os
import requests
from s3fs import S3FileSystem
import hashlib

test_bucket = "test"
files = [
    "test/test.txt",
    "test/subdir/test2.txt",
    "test/subdir2/test3.txt",
]
ip_address = "127.0.0.1"
port = 5555
endpoint_url = f"http://{ip_address}:{port}"


@pytest.fixture(scope="module")
def s3_base():
    # We create this module-level fixture to ensure that the server is only started once
    s3_server = ThreadedMotoServer(ip_address=ip_address, port=port)
    s3_server.start()
    if "AWS_ACCESS_KEY_ID" not in os.environ:
        os.environ["AWS_ACCESS_KEY_ID"] = "test"
    if "AWS_SECRET_ACCESS_KEY" not in os.environ:
        os.environ["AWS_SECRET_ACCESS_KEY"] = "test"
    yield
    s3_server.stop()


@pytest.fixture()
def init_s3_files(s3_base):
    requests.post(f"{endpoint_url}/moto-api/reset")
    s3fs = S3FileSystem(
        endpoint_url=endpoint_url,
    )
    s3fs.mkdir(test_bucket)
    s3fs.mkdir(f"{test_bucket}/subdir")
    s3fs.mkdir(f"{test_bucket}/subdir2")
    for file in files:
        with s3fs.open(file, "w") as f:
            f.write(f"test file: {file}")


def test_class():
    names_of_base_classes = [b.__name__ for b in S3Reader.__mro__]
    assert BasePydanticReader.__name__ in names_of_base_classes


def test_load_all_files(init_s3_files):
    reader = S3Reader(
        bucket=test_bucket,
        s3_endpoint_url=endpoint_url,
    )
    files = reader.list_resources()
    assert len(files) == 3
    documents = reader.load_data()
    assert len(documents) == len(files)


def test_load_single_file(init_s3_files):
    reader = S3Reader(
        bucket=test_bucket,
        key="test.txt",
        s3_endpoint_url=endpoint_url,
    )
    files = reader.list_resources()
    assert len(files) == 1
    documents = reader.load_data()
    assert len(documents) == 1
    assert documents[0].id_ == f"{endpoint_url}_{test_bucket}/test.txt"


def test_load_with_prefix(init_s3_files):
    reader = S3Reader(
        bucket=test_bucket,
        prefix="subdir",
        s3_endpoint_url=endpoint_url,
    )
    files = reader.list_resources()
    assert len(files) == 1
    assert str(files[0]).startswith(f"{test_bucket}/subdir")
    documents = reader.load_data()
    assert len(documents) == 1
    assert documents[0].id_ == f"{endpoint_url}_{test_bucket}/subdir/test2.txt"

    reader.prefix = "subdir2"
    files = reader.list_resources()
    assert len(files) == 1
    assert str(files[0]).startswith(f"{test_bucket}/subdir2")
    documents = reader.load_data()
    assert len(documents) == 1
    assert documents[0].id_ == f"{endpoint_url}_{test_bucket}/subdir2/test3.txt"


def test_load_not_recursive(init_s3_files):
    reader = S3Reader(
        bucket=test_bucket,
        recursive=False,
        s3_endpoint_url=endpoint_url,
    )
    documents = reader.load_data()
    assert len(documents) == 1
    assert documents[0].id_ == f"{endpoint_url}_{test_bucket}/test.txt"


def _compare_document_lists(
    documents1: List[Document], documents2: List[Document]
) -> None:
    assert len(documents1) == len(documents2)
    hashes_1 = {doc.hash for doc in documents1}
    hashes_2 = {doc.hash for doc in documents2}
    assert hashes_1 == hashes_2


def test_list_and_read_file_workflow(init_s3_files):
    reader = S3Reader(
        bucket=test_bucket,
        s3_endpoint_url=endpoint_url,
    )

    original_docs = reader.load_data()
    files = reader.list_resources()
    new_docs: List[Document] = []
    for file in files:
        file_info = reader.get_resource_info(file)
        assert file_info is not None
        assert len(file_info) == 4
        new_docs.extend(reader.load_resource(file))
    _compare_document_lists(original_docs, new_docs)

    new_docs = reader.load_resources(files)
    _compare_document_lists(original_docs, new_docs)


def test_read_file_content(init_s3_files):
    s3fs = S3FileSystem(
        endpoint_url=endpoint_url,
    )
    checksums = {}
    for file in files:
        with s3fs.open(file, "rb") as f:
            content = f.read()
            checksums[file] = hashlib.md5(content).hexdigest()

    reader = S3Reader(
        bucket=test_bucket,
        s3_endpoint_url=endpoint_url,
    )

    for file in files:
        content = reader.read_file_content(file)
        checksum = hashlib.md5(content).hexdigest()
        assert checksum == checksums[file]


def test_serialize():
    reader = S3Reader(
        bucket=test_bucket,
        s3_endpoint_url=endpoint_url,
    )

    schema = reader.schema()
    assert schema is not None
    assert len(schema) > 0
    assert "bucket" in schema["properties"]

    json = reader.json(exclude_unset=True)

    new_reader = S3Reader.parse_raw(json)
    assert new_reader.bucket == reader.bucket
    assert new_reader.s3_endpoint_url == reader.s3_endpoint_url
