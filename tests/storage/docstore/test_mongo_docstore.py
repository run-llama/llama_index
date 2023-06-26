from typing import List

import pytest

from llama_index.storage.docstore.mongo_docstore import MongoDocumentStore
from llama_index.schema import Document
from llama_index.schema import BaseNode
from llama_index.storage.kvstore.mongodb_kvstore import MongoDBKVStore

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None  # type: ignore


@pytest.fixture
def documents() -> List[Document]:
    return [
        Document(text="doc_1"),
        Document(text="doc_2"),
    ]


@pytest.fixture()
def mongodb_docstore(mongo_kvstore: MongoDBKVStore) -> MongoDocumentStore:
    return MongoDocumentStore(mongo_kvstore=mongo_kvstore)


@pytest.mark.skipif(MongoClient is None, reason="pymongo not installed")
def test_mongo_docstore(
    mongodb_docstore: MongoDocumentStore, documents: List[Document]
) -> None:
    ds = mongodb_docstore
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
    assert documents[0].get_content() == doc0.get_content()

    # test deleting documents
    ds.delete_document(documents[0].get_doc_id())
    assert len(ds.docs) == 1


@pytest.mark.skipif(MongoClient is None, reason="pymongo not installed")
def test_mongo_docstore_hash(
    mongodb_docstore: MongoDocumentStore, documents: List[Document]
) -> None:
    ds = mongodb_docstore

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
