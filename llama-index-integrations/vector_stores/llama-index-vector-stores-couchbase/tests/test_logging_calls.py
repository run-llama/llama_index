"""Offline tests: debug log calls must format cleanly."""

import logging
from typing import Any, List
from unittest.mock import MagicMock

import pytest
from llama_index.core.schema import TextNode
from llama_index.vector_stores.couchbase import CouchbaseSearchVectorStore


class _RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.messages: List[str] = []
        self.format_errors: List[Exception] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.messages.append(record.getMessage())
        except Exception as e:
            self.format_errors.append(e)


@pytest.fixture()
def handler() -> Any:
    logger = logging.getLogger("llama_index.vector_stores.couchbase.base")
    recorder = _RecordingHandler()
    previous_level = logger.level
    logger.addHandler(recorder)
    logger.setLevel(logging.DEBUG)
    yield recorder
    logger.removeHandler(recorder)
    logger.setLevel(previous_level)


@pytest.fixture()
def store() -> CouchbaseSearchVectorStore:
    store = CouchbaseSearchVectorStore.model_construct()
    store._cluster = MagicMock()
    store._collection = MagicMock()
    store._bucket_name = "bucket"
    store._text_key = "text"
    store._embedding_key = "embedding"
    store._metadata_key = "metadata"
    return store


def test_bucket_check_logs_the_error(
    store: CouchbaseSearchVectorStore, handler: _RecordingHandler
) -> None:
    store._cluster.buckets.return_value.get_bucket.side_effect = Exception("boom")

    assert store._check_bucket_exists() is False

    assert handler.format_errors == []
    assert any("boom" in message for message in handler.messages)


def test_add_logs_the_batch(
    store: CouchbaseSearchVectorStore, handler: _RecordingHandler
) -> None:
    store._collection.upsert_multi.return_value.all_ok = True
    node = TextNode(id_="n1", text="hello", embedding=[0.1, 0.2])

    assert store.add([node]) == ["n1"]

    assert handler.format_errors == []
    assert any("n1" in message for message in handler.messages)
