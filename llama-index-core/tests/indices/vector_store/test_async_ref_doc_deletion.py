import asyncio

import pytest

from llama_index.core import MockEmbedding, StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.docstore.types import RefDocInfo


class YieldingDocumentStore(SimpleDocumentStore):
    async def aget_ref_doc_info(self, ref_doc_id: str) -> RefDocInfo | None:
        # Let a concurrent document deletion finish before this lookup reads.
        await asyncio.sleep(0)
        return self.get_ref_doc_info(ref_doc_id)

    async def adelete_ref_doc(self, ref_doc_id: str, raise_error: bool = True) -> None:
        self.delete_ref_doc(ref_doc_id, raise_error=raise_error)


@pytest.mark.asyncio
async def test_adelete_ref_doc_cleans_up_index_before_deleting_docstore() -> None:
    docstore = YieldingDocumentStore()
    storage = StorageContext.from_defaults(docstore=docstore)
    node = TextNode(
        id_="chunk",
        text="Indexed text",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="document")},
    )
    index = VectorStoreIndex(
        [node], storage_context=storage, embed_model=MockEmbedding(embed_dim=2)
    )

    await index.adelete_ref_doc("document", delete_from_docstore=True)

    assert docstore.get_ref_doc_info("document") is None
    assert not docstore.document_exists("chunk")
    assert await index.as_retriever().aretrieve("Indexed text") == []
    assert index.ref_doc_info == {}
