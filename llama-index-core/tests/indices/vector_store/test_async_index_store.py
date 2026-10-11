from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.schema import Document


@pytest.mark.asyncio
async def test_adelete_nodes_uses_async_index_store(mock_embed_model) -> None:
    index = VectorStoreIndex.from_documents(
        [Document(text="Hello world.", id_="my-doc-id")],
        embed_model=mock_embed_model,
    )
    ref_doc_info = index.docstore.get_ref_doc_info("my-doc-id")
    assert ref_doc_info is not None

    index_store = index._storage_context.index_store
    index_store.add_index_struct = MagicMock(
        side_effect=AssertionError("sync index-store method called")
    )
    index_store.async_add_index_struct = AsyncMock()

    await index.adelete_nodes(ref_doc_info.node_ids, delete_from_docstore=True)

    index_store.async_add_index_struct.assert_awaited_once_with(index.index_struct)


@pytest.mark.asyncio
async def test_adelete_ref_doc_uses_async_index_store(mock_embed_model) -> None:
    index = VectorStoreIndex.from_documents(
        [Document(text="Hello world.", id_="my-doc-id")],
        embed_model=mock_embed_model,
    )
    index_store = index._storage_context.index_store
    index_store.add_index_struct = MagicMock(
        side_effect=AssertionError("sync index-store method called")
    )
    index_store.async_add_index_struct = AsyncMock()

    await index.adelete_ref_doc("my-doc-id")

    index_store.async_add_index_struct.assert_awaited_once_with(index.index_struct)
