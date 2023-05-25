from typing import Generator, List

import pytest
from pytest import MonkeyPatch

from llama_index.data_structs.node import Node
from llama_index.readers.schema.base import Document
from llama_index.schema import BaseDocument
from llama_index.storage.docstore.dynamodb_docstore import DynamoDBDocumentStore
from llama_index.storage.kvstore.dynamodb_kvstore import DynamoDBKVStore

try:
    import boto3
    from moto import mock_dynamodb

    has_boto_libs = True
except ImportError:
    has_boto_libs = False


@pytest.fixture()
def documents() -> List[Document]:
    return [Document("doc_1"), Document("doc_2")]


@pytest.fixture()
def kvstore_from_mocked_table(
    monkeypatch: MonkeyPatch,
) -> Generator[DynamoDBKVStore, None, None]:
    monkeypatch.setenv("MOTO_ALLOW_NONEXISTENT_REGION", "True")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "Andes")

    table_name = "test_table"
    with mock_dynamodb():
        client = boto3.client("dynamodb")
        client.create_table(
            TableName=table_name,
            AttributeDefinitions=[
                {"AttributeName": "collection", "AttributeType": "S"},
                {"AttributeName": "key", "AttributeType": "S"},
            ],
            KeySchema=[
                {"AttributeName": "collection", "KeyType": "HASH"},
                {"AttributeName": "key", "KeyType": "RANGE"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        yield DynamoDBKVStore.from_table_name(table_name)


@pytest.fixture()
def ddb_docstore(kvstore_from_mocked_table: DynamoDBKVStore) -> DynamoDBDocumentStore:
    return DynamoDBDocumentStore(dynamodb_kvstore=kvstore_from_mocked_table)


@pytest.mark.skipif(not has_boto_libs, reason="boto3 and/or moto not installed")
def test_docstore(ddb_docstore: DynamoDBDocumentStore) -> None:
    """Test docstore."""
    doc = Document("hello world", doc_id="d1", extra_info={"foo": "bar"})
    node = Node("my node", doc_id="d2", node_info={"node": "info"})

    # test get document
    docstore = ddb_docstore
    docstore.add_documents([doc, node])
    gd1 = docstore.get_document("d1")
    assert gd1 == doc
    gd2 = docstore.get_document("d2")
    assert gd2 == node


@pytest.mark.skipif(not has_boto_libs, reason="boto3 and/or moto not installed")
def test_dynamodb_docstore(
    ddb_docstore: DynamoDBDocumentStore, documents: List[Document]
) -> None:
    ds = ddb_docstore
    assert len(ds.docs) == 0

    # test adding documents
    ds.add_documents(documents)
    assert len(ds.docs) == 2
    assert all(isinstance(doc, BaseDocument) for doc in ds.docs.values())

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


@pytest.mark.skipif(not has_boto_libs, reason="boto3 and/or moto not installed")
def test_dynamodb_docstore_hash(
    ddb_docstore: DynamoDBDocumentStore, documents: List[Document]
) -> None:
    ds = ddb_docstore

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
