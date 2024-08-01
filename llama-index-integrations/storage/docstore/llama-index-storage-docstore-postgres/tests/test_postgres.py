from typing import List

import pytest
from llama_index.core.schema import BaseNode, Document
from llama_index.storage.docstore.postgres import (
    PostgresDocumentStore,
)
from llama_index.storage.kvstore.postgres import PostgresKVStore

try:
    import asyncpg  # noqa
    import psycopg2  # noqa
    import sqlalchemy  # noqa

    no_packages = False
except ImportError:
    no_packages = True


@pytest.fixture()
def documents() -> List[Document]:
    return [
        Document(text="doc_1"),
        Document(text="doc_2"),
    ]


@pytest.fixture()
def postgres_docstore(postgres_kvstore: PostgresKVStore) -> PostgresDocumentStore:
    return PostgresDocumentStore(postgres_kvstore=postgres_kvstore)


@pytest.mark.skipif(
    no_packages, reason="ayncpg, pscopg2-binary and sqlalchemy not installed"
)
def test_postgres_docstore(
    postgres_docstore: PostgresDocumentStore, documents: List[Document]
) -> None:
    ds = postgres_docstore
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


@pytest.mark.skipif(
    no_packages, reason="ayncpg, pscopg2-binary and sqlalchemy not installed"
)
def test_postgres_docstore_hash(
    postgres_docstore: PostgresDocumentStore, documents: List[Document]
) -> None:
    ds = postgres_docstore

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
