import pytest
import shutil
import os
import requests
from lancedb import AsyncConnection, DBConnection
from lancedb.table import AsyncTable, Table
import pandas as pd

from llama_index.indices.managed.lancedb.retriever import LanceDBRetriever
from llama_index.indices.managed.lancedb.query_engine import LanceDBRetrieverQueryEngine
from llama_index.indices.managed.lancedb import LanceDBMultiModalIndex
from llama_index.indices.managed.lancedb.utils import (
    LocalConnectionConfig,
    TableConfig,
    EmbeddingConfig,
    IndexingConfig,
    LanceDBTextModel,
    LanceDBMultiModalModel,
)
from llama_index.core.schema import Document, NodeWithScore
from llama_index.core import Settings
from llama_index.core.llms import MockLLM
from typing import List


@pytest.fixture()
def document_data() -> List[Document]:
    return [Document(id_="1", text="Hello"), Document(id_="2", text="Test")]


@pytest.fixture()
def data() -> pd.DataFrame:
    labels = ["cat", "cat", "dog", "dog", "horse", "horse"]
    uris = [
        "http://farm1.staticflickr.com/53/167798175_7c7845bbbd_z.jpg",
        "http://farm1.staticflickr.com/134/332220238_da527d8140_z.jpg",
        "http://farm9.staticflickr.com/8387/8602747737_2e5c2a45d4_z.jpg",
        "http://farm5.staticflickr.com/4092/5017326486_1f46057f5f_z.jpg",
        "http://farm9.staticflickr.com/8216/8434969557_d37882c42d_z.jpg",
        "http://farm6.staticflickr.com/5142/5835678453_4f3a4edb45_z.jpg",
    ]
    ids = [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
    ]
    image_bytes = [requests.get(uri).content for uri in uris]

    return pd.DataFrame(
        {"id": ids, "label": labels, "image_uri": uris, "image_bytes": image_bytes}
    )


@pytest.mark.asyncio
async def test_init(document_data: List[Document], data: List[dict]) -> None:
    if os.path.exists("lancedb"):
        shutil.rmtree("lancedb")

    first = LanceDBMultiModalIndex(
        uri="lancedb/data",
        text_embedding_model="sentence-transformers",
        embedding_model_kwargs={"name": "all-MiniLM-L6-v2"},
        table_name="test_table",
    )
    assert first.connection_config == LocalConnectionConfig(
        uri="lancedb/data", use_async=False
    )
    assert first.table_config == TableConfig(
        table_name="test_table", table_exists=False
    )
    assert first.indexing_config == IndexingConfig(
        indexing="IVF_PQ", indexing_kwargs={}
    )
    assert first.embedding_config == EmbeddingConfig(
        text_embedding_model="sentence-transformers",
        multi_modal_embedding_model=None,
        embedding_kwargs={"name": "all-MiniLM-L6-v2"},
    )
    second = await LanceDBMultiModalIndex.from_documents(
        documents=document_data,
        uri="lancedb/documents",
        text_embedding_model="sentence-transformers",
        embedding_model_kwargs={"name": "all-MiniLM-L6-v2"},
        table_name="test_table",
        indexing="NO_INDEXING",
        use_async=True,
    )
    assert isinstance(second._connection, AsyncConnection)
    assert isinstance(second._table, AsyncTable)
    assert isinstance(second._embedding_model, LanceDBTextModel)
    third = await LanceDBMultiModalIndex.from_data(
        data=data,
        uri="lancedb/from-data",
        multimodal_embedding_model="open-clip",
        indexing="NO_INDEXING",
        use_async=False,
    )
    assert isinstance(third._connection, DBConnection)
    assert isinstance(third._table, Table)
    assert isinstance(third._embedding_model, LanceDBMultiModalModel)
    shutil.rmtree("lancedb")


@pytest.mark.asyncio
async def test_retriever_qe(document_data: List[Document]) -> None:
    Settings.llm = MockLLM()
    second = await LanceDBMultiModalIndex.from_documents(
        documents=document_data,
        uri="lancedb/documents",
        text_embedding_model="sentence-transformers",
        embedding_model_kwargs={"name": "all-MiniLM-L6-v2"},
        table_name="test_table",
        indexing="NO_INDEXING",
        use_async=True,
    )
    retr = second.as_retriever()
    assert isinstance(retr, LanceDBRetriever)
    retrieved = await retr.aretrieve(query_str="Hello")
    assert isinstance(retrieved, list)
    assert len(retrieved) > 0
    assert isinstance(retrieved[0], NodeWithScore)
    qe = second.as_query_engine()
    assert isinstance(qe, LanceDBRetrieverQueryEngine)
    response = await qe.aquery(query_str="Hello")
    assert isinstance(response.response, str)
    shutil.rmtree("lancedb")
