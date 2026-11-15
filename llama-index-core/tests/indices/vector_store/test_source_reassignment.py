from pathlib import Path

import pytest
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import (
    NodeRelationship,
    QueryBundle,
    RelatedNodeInfo,
    TextNode,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_reassigned_node_survives_source_deletion(
    use_async: bool, tmp_path: Path
) -> None:
    embed_model = MockEmbedding(embed_dim=2)
    nodes = [
        TextNode(
            id_=node_id,
            text=node_id,
            embedding=[1.0, 0.0],
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="old")},
        )
        for node_id in ["chunk", "sibling"]
    ]
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    updated = TextNode(
        id_="chunk",
        text="updated content",
        embedding=[0.0, 1.0],
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="new")},
    )
    if use_async:
        await index.ainsert_nodes([updated])
    else:
        index.insert_nodes([updated])

    index.storage_context.persist(persist_dir=str(tmp_path))
    restored = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=str(tmp_path)), embed_model=embed_model
    )
    restored.delete_ref_doc("old", delete_from_docstore=True)

    results = restored.as_retriever().retrieve(
        QueryBundle(query_str="find", embedding=[0.0, 1.0])
    )
    assert [(result.node.node_id, result.node.text) for result in results] == [
        ("chunk", "updated content")
    ]
    assert restored.ref_doc_info["new"].node_ids == ["chunk"]
