from typing import Any
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.lancedb.base import TableNotFoundError, VectorStoreQuery
import pytest
import pytest
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.embeddings.mock_embed_model import MockEmbedding, BaseEmbedding
from llama_index.core.settings import Settings
from pathlib import Path

try:
    from lancedb.pydantic import Vector, LanceModel
    import lancedb

    deps = True
except ImportError:
    deps = None


class TestModel(LanceModel):
    text: str
    id_: str
    vector: Vector(dim=3)


class TmpMockEmbedding(MockEmbedding):
    async def _aget_text_embedding(self, text: str) -> list[float]:
        if text == "test1":
            return [0.1, 0.2, 0.3]
        elif text == "test2":
            return [0.4, 0.5, 0.7]
        elif text == "test3":
            return [0.6, 0.2, 0.1]
        return self._get_vector()

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await self._aget_text_embedding(text=query)

    def _get_query_embedding(self, query: str) -> list[float]:
        """Get query embedding."""
        return self._get_text_embedding(text=query)

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get text embedding."""
        if text == "test1":
            return [0.1, 0.2, 0.3]
        elif text == "test2":
            return [0.4, 0.5, 0.6]
        elif text == "test3":
            return [0.6, 0.2, 0.1]
        return self._get_vector()


@pytest.fixture(scope="module")
def embed_model() -> BaseEmbedding:
    embed_model = TmpMockEmbedding(embed_dim=3)
    embed_model.callback_manager = Settings.callback_manager
    return embed_model


@pytest.fixture(scope="module")
def text_node_list(embed_model) -> list[TextNode]:
    nodes = [
        TextNode(
            text="test1",
            id_="11111111-1111-1111-1111-111111111111",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-0")},
        ),
        TextNode(
            text="test2",
            id_="22222222-2222-2222-2222-222222222222",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-1")},
        ),
        TextNode(
            text="test3",
            id_="33333333-3333-3333-3333-333333333333",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-2")},
        ),
    ]

    for node in nodes:
        node_embedding = embed_model.get_text_embedding(node.text)
        node.embedding = node_embedding
    return nodes


def test_class():
    names_of_base_classes = [b.__name__ for b in LanceDBVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


@pytest.mark.skipif(
    deps is None,
    reason="Need to install lancedb locally to run this test.",
)
@pytest.mark.parametrize("mode", ["overwrite", "create"])
def test_vector_store_init_create_pass(tmp_path: Path, mode: str) -> None:
    # given
    # when
    vector_store = LanceDBVectorStore(uri=str(tmp_path / "test_lancedb"), mode=mode)

    # then
    assert vector_store._table is None
    with pytest.raises(TableNotFoundError):
        vector_store.table


@pytest.mark.skipif(
    deps is None,
    reason="Need to install lancedb locally to run this test.",
)
@pytest.mark.parametrize("mode", ["overwrite", "create", "append"])
def test_vector_store_init_table_exists(tmp_path: Path, mode: str) -> None:
    # given
    connection = lancedb.connect(str(tmp_path / "test_lancedb"))
    connection.create_table(name="test_table", schema=TestModel)

    # when
    vector_store = LanceDBVectorStore(
        mode=mode, table_name="test_table", connection=connection
    )

    # then
    assert isinstance(vector_store._table, lancedb.db.LanceTable)
    assert isinstance(vector_store.table, lancedb.db.LanceTable)


@pytest.mark.skipif(
    deps is None,
    reason="Need to install lancedb locally to run this test.",
)
def test_vector_store_init_append_error(tmp_path: Path) -> None:
    # given
    connection = lancedb.connect(str(tmp_path / "test_lancedb"))

    # when & then
    with pytest.raises(TableNotFoundError):
        LanceDBVectorStore(
            mode="append", table_name="test_table", connection=connection
        )


@pytest.mark.skipif(
    deps is None,
    reason="Need to install lancedb locally to run this test.",
)
def test_vector_store_from_table(tmp_path: Path) -> None:
    # given
    connection = lancedb.connect(str(tmp_path / "test_lancedb"))
    table = connection.create_table(name="test_table", schema=TestModel)

    # when
    vector_store = LanceDBVectorStore.from_table(table=table)

    # then
    assert isinstance(vector_store, LanceDBVectorStore)
    assert isinstance(vector_store._table, lancedb.db.LanceTable)
    assert vector_store._table.name == table.name
    assert connection == table._conn


@pytest.mark.skipif(
    deps is None,
    reason="Need to install lancedb locally to run this test.",
)
def test_table_exists(tmp_path: Path) -> None:
    # given
    connection = lancedb.connect(str(tmp_path / "test_lancedb"))
    table = connection.create_table(name="test_table", schema=TestModel)

    # when
    vector_store = LanceDBVectorStore.from_table(table=table)

    # then
    assert vector_store._table_exists(tbl_name=table.name)
    assert not vector_store._table_exists(tbl_name="non_existent_table")


@pytest.mark.skipif(
    deps is None,
    reason="Need to install lancedb locally to run this test.",
)
def test_create_index_pass(tmp_path: Path, text_node_list: list[TextNode]) -> None:
    # given
    vector_store = LanceDBVectorStore(
        uri=str(tmp_path / "test_lancedb"), mode="overwrite"
    )
    vector_store.add(text_node_list)

    # when
    vector_store.create_index(num_partitions=2, index_type="IVF_FLAT", sample_rate=2)

    # then
    assert isinstance(vector_store._table, lancedb.db.LanceTable)
    assert vector_store._table.list_indices()


@pytest.mark.skipif(
    deps is None,
    reason="Need to install lancedb locally to run this test.",
)
def test_add(tmp_path: Path, text_node_list: list[TextNode]) -> None:
    # given
    vector_store = LanceDBVectorStore(
        uri=str(tmp_path / "test_lancedb"), mode="overwrite"
    )

    # when
    vector_store.add(text_node_list)

    # then
    assert vector_store._table_exists()
    assert vector_store._table.count_rows() == len(text_node_list)


@pytest.mark.skipif(
    deps is None,
    reason="Need to install lancedb locally to run this test.",
)
def test_delete(tmp_path: Path, text_node_list: list[TextNode]) -> None:
    # given
    vector_store = LanceDBVectorStore(
        uri=str(tmp_path / "test_lancedb"), mode="overwrite"
    )
    vector_store.add(text_node_list)

    # when
    vector_store.delete(ref_doc_id="test-0")

    # then
    assert vector_store._table.count_rows() == len(text_node_list) - 1


@pytest.mark.skipif(
    deps is None,
    reason="Need to install lancedb locally to run this test.",
)
def test_delete_nodes(tmp_path: Path, text_node_list: list[TextNode]) -> None:
    # given
    vector_store = LanceDBVectorStore(
        uri=str(tmp_path / "test_lancedb"), mode="overwrite"
    )
    vector_store.add(text_node_list)

    # when
    vector_store.delete_nodes(node_ids=[text_node_list[0].id_, text_node_list[1].id_])

    # then
    assert vector_store._table.count_rows() == len(text_node_list) - 2


@pytest.mark.skipif(
    deps is None,
    reason="Need to install lancedb locally to run this test.",
)
def test_get_nodes(tmp_path: Path, text_node_list: list[TextNode]) -> None:
    # given
    vector_store = LanceDBVectorStore(
        uri=str(tmp_path / "test_lancedb"), mode="overwrite"
    )
    vector_store.add(text_node_list)

    # when
    retieved_nodes = vector_store.get_nodes(
        node_ids=[text_node_list[0].id_, text_node_list[1].id_]
    )

    # then
    assert len(retieved_nodes) == 2
    assert retieved_nodes[0].id_ == text_node_list[0].id_
    assert retieved_nodes[1].id_ == text_node_list[1].id_


@pytest.mark.skipif(
    deps is None,
    reason="Need to install lancedb locally to run this test.",
)
def test_vector_query(
    tmp_path: Path, text_node_list: list[TextNode], embed_model
) -> None:
    # given
    vector_store = LanceDBVectorStore(
        uri=str(tmp_path / "test_lancedb"), mode="overwrite"
    )
    vector_store.add(text_node_list)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    retriever = index.as_retriever(similarity_top_k=1)

    # when
    response = retriever.retrieve("test1")

    # then
    assert len(response) == 1
    assert response[0].id_ == "11111111-1111-1111-1111-111111111111"


@pytest.mark.skipif(
    deps is None,
    reason="Need to install lancedb and huggingface locally to run this test.",
)
def test_fts_query(tmp_path: Path, text_node_list: list[TextNode], embed_model) -> None:
    # given
    vector_store = LanceDBVectorStore(
        uri=str(tmp_path / "test_lancedb"), mode="overwrite"
    )
    vector_store.add(text_node_list)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    # when
    response = index.as_retriever(vector_store_kwargs={"query_type": "fts"}).retrieve(
        "test1"
    )

    # then
    assert len(response) == 1
    assert response[0].id_ == "11111111-1111-1111-1111-111111111111"


@pytest.mark.skipif(
    deps is None,
    reason="Need to install lancedb and huggingface locally to run this test.",
)
def test_hybrid_query(
    tmp_path: Path, text_node_list: list[TextNode], embed_model
) -> None:
    # given
    vector_store = LanceDBVectorStore(
        uri=str(tmp_path / "test_lancedb"), mode="overwrite"
    )
    vector_store.add(text_node_list)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    # when
    response = index.as_retriever(
        vector_store_kwargs={"query_type": "hybrid"}, similarity_top_k=3
    ).retrieve("test1")

    # then
    assert len(response) == 3
    assert response[0].id_ == "11111111-1111-1111-1111-111111111111"


@pytest.mark.skipif(
    deps is None,
    reason="Need to install lancedb locally to run this test.",
)
@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("create_index", {}),
        ("delete", {"ref_doc_id": "test-0"}),
        ("delete_nodes", {"node_ids": []}),
        ("get_nodes", {}),
        ("query", {"query": VectorStoreQuery()}),
    ],
)
def test_method_table_error(
    tmp_path: Path, method: str, kwargs: dict[str, Any]
) -> None:
    # given
    vector_store = LanceDBVectorStore(
        uri=str(tmp_path / "test_lancedb"), mode="overwrite"
    )

    # when
    with pytest.raises(TableNotFoundError):
        getattr(vector_store, method)(**kwargs)
