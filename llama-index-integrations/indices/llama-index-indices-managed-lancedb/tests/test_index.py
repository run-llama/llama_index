import pytest
import shutil
import os
import requests
import uuid
from lancedb import AsyncConnection, DBConnection
from lancedb.table import AsyncTable, Table
from typing import Generator
import pandas as pd

from llama_index.indices.managed.lancedb.retriever import LanceDBRetriever
from llama_index.indices.managed.lancedb.query_engine import LanceDBRetrieverQueryEngine
from llama_index.indices.managed.lancedb import LanceDBMultiModalIndex
from llama_index.indices.managed.lancedb.utils import (
    aquery_multimodal,
    aquery_text,
    query_multimodal,
    query_text,
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
from PIL import Image


@pytest.fixture()
def document_data() -> List[Document]:
    return [Document(id_="1", text="Hello"), Document(id_="2", text="Test")]


@pytest.fixture()
def data() -> pd.DataFrame:
    labels = ["cat", "cat", "dog", "dog", "horse", "horse"]
    uris = [
        "https://picsum.photos/200/200?random=1",
        "https://picsum.photos/200/200?random=2",
        "https://picsum.photos/200/200?random=3",
        "https://picsum.photos/200/200?random=4",
        "https://picsum.photos/200/200?random=5",
        "https://picsum.photos/200/200?random=6",
    ]
    ids = [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
    ]
    image_bytes = []
    for uri in uris:
        response = requests.get(uri)
        response.raise_for_status()
        image_bytes.append(response.content)

    metadata = [
        '{"mimetype": "image/jpeg"}',
        '{"mimetype": "image/jpeg"}',
        '{"mimetype": "image/jpeg"}',
        '{"mimetype": "image/jpeg"}',
        '{"mimetype": "image/jpeg"}',
        '{"mimetype": "image/jpeg"}',
    ]

    return pd.DataFrame(
        {
            "id": ids,
            "label": labels,
            "image_uri": uris,
            "image_bytes": image_bytes,
            "metadata": metadata,
        }
    )


@pytest.fixture()
def uri() -> Generator[str, None, None]:
    uri = f"lancedb/{uuid.uuid4()}"
    yield uri
    if os.path.exists(uri):
        shutil.rmtree(uri)


@pytest.mark.asyncio
async def test_init(document_data: List[Document], data: List[dict], uri: str) -> None:
    first = LanceDBMultiModalIndex(
        uri=uri,
        text_embedding_model="sentence-transformers",
        embedding_model_kwargs={"name": "all-MiniLM-L6-v2"},
        table_name="test_table",
    )
    assert first.connection_config.uri == uri
    assert first.connection_config.use_async is False

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
        uri=f"{uri}/documents",
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
        uri=f"{uri}/from-data",
        multimodal_embedding_model="open-clip",
        indexing="NO_INDEXING",
        use_async=False,
    )
    assert isinstance(third._connection, DBConnection)
    assert isinstance(third._table, Table)
    assert isinstance(third._embedding_model, LanceDBMultiModalModel)


@pytest.mark.asyncio
async def test_retriever_qe(uri: str, document_data: List[Document]) -> None:
    Settings.llm = MockLLM()
    second = await LanceDBMultiModalIndex.from_documents(
        documents=document_data,
        uri=f"{uri}/documents",
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


IMAGE_ROW = {
    "id": "img-1",
    "image_uri": "https://example.com/cat.png",
    "image_bytes": b"\x89PNG\r\n",
    "label": "cat",
    "metadata": "{}",
    "_distance": 0.42,
}


class _FakeQuery:
    """Stands in for a lancedb query builder; only `to_list` is used here."""

    def __init__(self, rows: List[dict]) -> None:
        self._rows = rows

    def to_list(self) -> List[dict]:
        return self._rows


class _FakeTable:
    def __init__(self, rows: List[dict]) -> None:
        self._rows = rows

    def search(self, *args: object, **kwargs: object) -> _FakeQuery:
        return _FakeQuery(self._rows)


class _FakeAsyncQuery:
    def __init__(self, rows: List[dict]) -> None:
        self._rows = rows

    async def to_list(self) -> List[dict]:
        return self._rows


class _FakeAsyncTable:
    def __init__(self, rows: List[dict]) -> None:
        self._rows = rows

    async def search(self, *args: object, **kwargs: object) -> _FakeAsyncQuery:
        return _FakeAsyncQuery(self._rows)


def _assert_image_result(retrieved: List[NodeWithScore]) -> None:
    assert len(retrieved) == 1
    assert isinstance(retrieved[0], NodeWithScore)
    assert retrieved[0].node.id_ == "img-1"
    assert retrieved[0].score == pytest.approx(0.42)


def test_query_multimodal_builds_nodes_from_image_rows() -> None:
    """`NodeWithScore` takes its node by keyword; a positional node is a TypeError."""
    _assert_image_result(
        query_multimodal(_FakeTable([IMAGE_ROW]), query=Image.new("RGB", (4, 4)))
    )


def test_query_text_builds_nodes_from_image_rows() -> None:
    """A row without a `text` column falls through to the image branch."""
    _assert_image_result(query_text(_FakeTable([IMAGE_ROW]), query="a cat"))


@pytest.mark.asyncio
async def test_aquery_multimodal_builds_nodes_from_image_rows() -> None:
    _assert_image_result(
        await aquery_multimodal(
            _FakeAsyncTable([IMAGE_ROW]), query=Image.new("RGB", (4, 4))
        )
    )


@pytest.mark.asyncio
async def test_aquery_text_builds_nodes_from_image_rows() -> None:
    _assert_image_result(await aquery_text(_FakeAsyncTable([IMAGE_ROW]), query="a cat"))
