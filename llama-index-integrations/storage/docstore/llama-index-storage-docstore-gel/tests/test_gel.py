import subprocess
import pytest
import os
from typing import List, Generator
from llama_index.core.schema import BaseNode, Document
from llama_index.storage.docstore.gel import (
    GelDocumentStore,
)
from llama_index.storage.kvstore.gel import GelKVStore

try:
    import gel  # noqa

    no_packages = False
except ImportError:
    no_packages = True

skip_in_cicd = os.environ.get("CI") is not None
try:
    if not skip_in_cicd:
        subprocess.run(["gel", "project", "init", "--non-interactive"], check=True)
except subprocess.CalledProcessError as e:
    print(e)


@pytest.fixture()
def documents() -> List[Document]:
    return [
        Document(text="doc_1"),
        Document(text="doc_2"),
    ]


@pytest.fixture()
def gel_kvstore() -> Generator[GelKVStore, None, None]:
    kvstore = None
    try:
        kvstore = GelKVStore()
        yield kvstore
    finally:
        if kvstore:
            keys = kvstore.get_all().keys()
            for key in keys:
                kvstore.delete(key)


@pytest.fixture()
def gel_docstore(gel_kvstore: GelKVStore) -> Generator[GelDocumentStore, None, None]:
    docstore = None
    try:
        docstore = GelDocumentStore(gel_kvstore=gel_kvstore)
        for id_ in docstore.docs:
            docstore.delete_document(id_)
        yield docstore
    finally:
        if docstore:
            for id_ in docstore.docs:
                docstore.delete_document(id_)


@pytest.mark.skipif(no_packages or skip_in_cicd, reason="gel not installed")
def test_gel_docstore(
    gel_docstore: GelDocumentStore, documents: List[Document]
) -> None:
    ds = gel_docstore
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


@pytest.mark.skipif(no_packages or skip_in_cicd, reason="gel not installed")
def test_gel_docstore_hash(
    gel_docstore: GelDocumentStore, documents: List[Document]
) -> None:
    ds = gel_docstore

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
