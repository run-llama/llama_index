from typing import List

import pytest

from llama_index.storage.docstore.redis_docstore import RedisDocumentStore
from llama_index.readers.schema.base import Document
from llama_index.schema import BaseNode
from llama_index.storage.kvstore.redis_kvstore import RedisKVStore

try:
    from redis import Redis
except ImportError:
    Redis = None  # type: ignore


@pytest.fixture
def documents() -> List[Document]:
    return [
        Document("doc_1"),
        Document("doc_2"),
    ]


@pytest.fixture()
def redis_docstore(redis_kvstore: RedisKVStore) -> RedisDocumentStore:
    return RedisDocumentStore(redis_kvstore=redis_kvstore)


@pytest.mark.skipif(Redis is None, reason="redis not installed")
def test_redis_docstore(
    redis_docstore: RedisDocumentStore, documents: List[Document]
) -> None:
    ds = redis_docstore
    assert len(ds.docs) == 0

    # test adding documents
    ds.add_documents(documents)
    assert len(ds.docs) == 2
    assert all(isinstance(doc, BaseNode) for doc in ds.docs.values())

    # test updating documents
    ds.add_documents(documents)
    print(ds.docs)
    assert len(ds.docs) == 2

    # test getting documents
    doc0 = ds.get_document(documents[0].get_doc_id())
    assert doc0 is not None
    assert documents[0].text == doc0.text

    # test deleting documents
    ds.delete_document(documents[0].get_doc_id())
    assert len(ds.docs) == 1


@pytest.mark.skipif(Redis is None, reason="redis not installed")
def test_redis_docstore_hash(
    redis_docstore: RedisDocumentStore, documents: List[Document]
) -> None:
    ds = redis_docstore

    # Test setting hash
    ds.set_document_hash("test_doc_id", "test_doc_hash")
    doc_hash = ds.get_document_hash("test_doc_id")
    assert doc_hash == "test_doc_hash"

    # Test updating hash
    ds.set_document_hash("test_doc_id", "test_doc_hash_new")
    doc_hash = ds.get_document_hash("test_doc_id")
    assert doc_hash == "test_doc_hash_new"

    # Test getting non-existent
    doc_hash = ds.get_document_hash("test_not_exist")
    assert doc_hash is None


@pytest.mark.skipif(Redis is None, reason="redis not installed")
def test_redis_docstore_deserialization(
    redis_docstore: RedisDocumentStore, documents: List[Document]
) -> None:
    from llama_index import (
        ListIndex,
        StorageContext,
        Document,
    )
    from llama_index.storage.docstore import RedisDocumentStore
    from llama_index.storage.index_store import RedisIndexStore

    ds = RedisDocumentStore.from_host_and_port(
        "127.0.0.1", int(6379), namespace="data4"
    )
    idxs = RedisIndexStore.from_host_and_port("127.0.0.1", int(6379), namespace="data4")

    storage_context = StorageContext.from_defaults(docstore=ds, index_store=idxs)

    index = ListIndex.from_documents(
        [Document("hello world2")], storage_context=storage_context
    )
    # fails here
    doc = index.docstore.docs
    print(doc)
