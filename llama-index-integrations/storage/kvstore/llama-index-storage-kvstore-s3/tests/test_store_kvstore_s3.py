import io
import json

from llama_index.core.storage.kvstore.types import BaseKVStore
from llama_index.storage.kvstore.s3 import S3DBKVStore


class _FakeObject:
    """Stand-in for a boto3 S3 object summary."""

    def __init__(self, key: str, body: bytes) -> None:
        self.key = key
        self._body = body

    def get(self) -> dict:
        return {"Body": io.BytesIO(self._body)}

    def delete(self) -> None:
        raise AssertionError("delete is not exercised by these tests")


class _FakeObjectList(list):
    def limit(self, count: int) -> list:
        return self[:count]


class _FakeObjects:
    def __init__(self, bucket: "_FakeBucket") -> None:
        self._bucket = bucket

    def filter(self, Prefix: str = "") -> _FakeObjectList:
        return _FakeObjectList(
            [
                _FakeObject(key, body)
                for key, body in sorted(self._bucket.storage.items())
                if key.startswith(Prefix)
            ]
        )


class _FakeBucket:
    """Minimal stand-in for ``boto3.Bucket``: no network, no credentials."""

    def __init__(self) -> None:
        self.storage: dict = {}
        self.objects = _FakeObjects(self)

    def put_object(self, Key: str, Body: str) -> None:
        self.storage[Key] = Body.encode("utf-8") if isinstance(Body, str) else Body


def test_class():
    names_of_base_classes = [b.__name__ for b in S3DBKVStore.__mro__]
    assert BaseKVStore.__name__ in names_of_base_classes


def test_object_keys_use_s3_separators() -> None:
    store = S3DBKVStore(bucket=_FakeBucket(), path="kv")

    assert store._get_object_key("notes", "n1") == "kv/notes/n1.json"


def test_get_all_ignores_collections_sharing_the_prefix() -> None:
    bucket = _FakeBucket()
    store = S3DBKVStore(bucket=bucket, path="kv")
    store.put("a", {"v": "A"}, collection="default")
    store.put("b", {"v": "B"}, collection="defaults")

    assert store.get_all("default") == {"a": {"v": "A"}}


def test_reads_keys_written_with_s3_separators() -> None:
    bucket = _FakeBucket()
    # Keys as written by any other S3 client, e.g. the AWS CLI.
    bucket.storage["kv/notes/n1.json"] = json.dumps({"t": "hello"}).encode("utf-8")
    store = S3DBKVStore(bucket=bucket, path="kv")

    assert store.get("n1", collection="notes") == {"t": "hello"}
    assert store.get_all("notes") == {"n1": {"t": "hello"}}
