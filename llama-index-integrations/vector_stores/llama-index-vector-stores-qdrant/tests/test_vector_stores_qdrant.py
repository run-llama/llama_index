from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
import pytest


def test_class():
    names_of_base_classes = [b.__name__ for b in QdrantVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def test_delete__and_get_nodes(vector_store: QdrantVectorStore) -> None:
    vector_store.delete_nodes(node_ids=["11111111-1111-1111-1111-111111111111"])

    existing_nodes = vector_store.get_nodes(
        node_ids=[
            "11111111-1111-1111-1111-111111111111",
            "22222222-2222-2222-2222-222222222222",
            "33333333-3333-3333-3333-333333333333",
        ]
    )
    assert len(existing_nodes) == 2


def test_clear(vector_store: QdrantVectorStore) -> None:
    vector_store.clear()
    with pytest.raises(ValueError, match="Collection test not found"):
        vector_store.get_nodes(
            node_ids=[
                "11111111-1111-1111-1111-111111111111",
                "22222222-2222-2222-2222-222222222222",
                "33333333-3333-3333-3333-333333333333",
            ]
        )


@pytest.mark.asyncio()
async def test_adelete_and_aget(vector_store: QdrantVectorStore) -> None:
    await vector_store.adelete_nodes(node_ids=["11111111-1111-1111-1111-111111111111"])

    existing_nodes = await vector_store.aget_nodes(
        node_ids=[
            "11111111-1111-1111-1111-111111111111",
            "22222222-2222-2222-2222-222222222222",
            "33333333-3333-3333-3333-333333333333",
        ]
    )
    assert len(existing_nodes) == 2


@pytest.mark.asyncio()
async def test_aclear(vector_store: QdrantVectorStore) -> None:
    await vector_store.aclear()
    with pytest.raises(ValueError, match="Collection test not found"):
        await vector_store.aget_nodes(
            node_ids=[
                "11111111-1111-1111-1111-111111111111",
                "22222222-2222-2222-2222-222222222222",
                "33333333-3333-3333-3333-333333333333",
            ]
        )
