import asyncio
from pathlib import Path
from typing import Any, List, Optional

import pytest

from llama_index.core.embeddings import MockEmbedding
from llama_index.core.indices.loading import load_index_from_storage
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.docstore.types import RefDocInfo
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.vector_stores.simple import SimpleVectorStore


@pytest.fixture
def nodes() -> List[TextNode]:
    return [
        TextNode(
            id_=node_id,
            text=node_id,
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id=ref_doc_id)
            },
        )
        for node_id, ref_doc_id in [
            ("target-a", "target"),
            ("target-b", "target"),
            ("keep", "other"),
        ]
    ]


@pytest.mark.asyncio
async def test_failed_vector_deletion_preserves_retrievable_nodes(
    nodes: List[TextNode],
) -> None:
    class FailingVectorStore(SimpleVectorStore):
        async def adelete(self, ref_doc_id: str, **kwargs: Any) -> None:
            raise RuntimeError("vector store unavailable")

    index = VectorStoreIndex(
        nodes,
        storage_context=StorageContext.from_defaults(vector_store=FailingVectorStore()),
        embed_model=MockEmbedding(embed_dim=4),
    )

    with pytest.raises(RuntimeError, match="vector store unavailable"):
        await index.adelete_ref_doc("target", delete_from_docstore=True)

    retrieved = await index.as_retriever(similarity_top_k=3).aretrieve("documents")
    assert {hit.node.node_id for hit in retrieved} == {"target-a", "target-b", "keep"}


@pytest.mark.asyncio
@pytest.mark.parametrize("delete_from_docstore", [False, True])
async def test_async_deletion_preserves_consistency_after_reload(
    nodes: List[TextNode], tmp_path: Path, delete_from_docstore: bool
) -> None:
    class YieldingDocumentStore(SimpleDocumentStore):
        async def aget_ref_doc_info(self, ref_doc_id: str) -> Optional[RefDocInfo]:
            await asyncio.sleep(0)
            return self.get_ref_doc_info(ref_doc_id)

        async def adelete_ref_doc(
            self, ref_doc_id: str, raise_error: bool = True
        ) -> None:
            # Model a store whose deletion completes before an in-flight read.
            self.delete_ref_doc(ref_doc_id, raise_error=raise_error)

    embed_model = MockEmbedding(embed_dim=4)
    index = VectorStoreIndex(
        nodes,
        storage_context=StorageContext.from_defaults(docstore=YieldingDocumentStore()),
        embed_model=embed_model,
    )

    await index.adelete_ref_doc("target", delete_from_docstore=delete_from_docstore)
    index.storage_context.persist(persist_dir=str(tmp_path))
    restored = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=str(tmp_path)), embed_model=embed_model
    )

    assert set(restored.ref_doc_info) == {"other"}
    retrieved = await restored.as_retriever(similarity_top_k=3).aretrieve("documents")
    assert [hit.node.node_id for hit in retrieved] == ["keep"]
    assert restored.docstore.document_exists("target-a") is not delete_from_docstore
    assert restored.docstore.document_exists("target-b") is not delete_from_docstore
