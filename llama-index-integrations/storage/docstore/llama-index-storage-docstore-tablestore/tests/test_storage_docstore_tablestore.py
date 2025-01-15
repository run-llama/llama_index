import os
from typing import List

import pytest
from llama_index.core import Document
from llama_index.core.schema import BaseNode
from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore

from llama_index.storage.docstore.tablestore import TablestoreDocumentStore


def test_class():
    names_of_base_classes = [b.__name__ for b in TablestoreDocumentStore.__mro__]
    assert KVDocumentStore.__name__ in names_of_base_classes
    assert TablestoreDocumentStore.__name__ in names_of_base_classes


@pytest.fixture()
def documents() -> List[Document]:
    return [
        Document(text="doc_1", id_="1", metadata={"key1": "value1"}),
        Document(text="doc_2", id_="2", metadata={"key2": "value2"}),
    ]


# noinspection DuplicatedCode
@pytest.fixture()
def tablestore_doc_store() -> TablestoreDocumentStore:
    end_point = os.getenv("tablestore_end_point")
    instance_name = os.getenv("tablestore_instance_name")
    access_key_id = os.getenv("tablestore_access_key_id")
    access_key_secret = os.getenv("tablestore_access_key_secret")
    if (
        end_point is None
        or instance_name is None
        or access_key_id is None
        or access_key_secret is None
    ):
        pytest.skip(
            "end_point is None or instance_name is None or "
            "access_key_id is None or access_key_secret is None"
        )

    # 1. create tablestore vector store
    store = TablestoreDocumentStore.from_config(
        endpoint=os.getenv("tablestore_end_point"),
        instance_name=os.getenv("tablestore_instance_name"),
        access_key_id=os.getenv("tablestore_access_key_id"),
        access_key_secret=os.getenv("tablestore_access_key_secret"),
    )
    store.clear_all()
    return store


# noinspection DuplicatedCode
def test_tablestore_doc_store(
    tablestore_doc_store: TablestoreDocumentStore, documents: List[Document]
) -> None:
    ds = tablestore_doc_store
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


# noinspection DuplicatedCode
def test_tablestore_hash(
    tablestore_doc_store: TablestoreDocumentStore, documents: List[Document]
) -> None:
    ds = tablestore_doc_store
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


def test_delete_all(
    tablestore_doc_store: TablestoreDocumentStore, documents: List[Document]
):
    ds = tablestore_doc_store
    ds.add_documents(documents)
    assert len(ds.docs) >= 2

    ds.clear_all()
    assert len(ds.docs) == 0
