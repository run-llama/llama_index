"""Test ref doc deletion on MultiModalVectorStoreIndex."""

import pytest
from llama_index.core import Document
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.embeddings.mock_embed_model import MockMultiModalEmbedding
from llama_index.core.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.core.schema import ImageNode, NodeRelationship, RelatedNodeInfo


def _build_index() -> MultiModalVectorStoreIndex:
    image_node = ImageNode(
        image_path="img.png",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="img_doc")},
    )
    index = MultiModalVectorStoreIndex.from_documents(
        [
            Document(text="text one", doc_id="txt_doc_1"),
            Document(text="text two", doc_id="txt_doc_2"),
        ],
        embed_model=MockEmbedding(embed_dim=8),
        image_embed_model=MockMultiModalEmbedding(embed_dim=8),
    )
    index.insert_nodes([image_node])
    return index


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("ref_doc_id", ["txt_doc_1", "img_doc"])
async def test_delete_ref_doc_cleans_index_struct_and_stores(
    use_async: bool, ref_doc_id: str
) -> None:
    index = _build_index()
    assert len(index.index_struct.nodes_dict) == 3

    if use_async:
        await index.adelete_ref_doc(ref_doc_id, delete_from_docstore=True)
    else:
        index.delete_ref_doc(ref_doc_id, delete_from_docstore=True)

    assert len(index.index_struct.nodes_dict) == 2
    assert ref_doc_id not in index.ref_doc_info
    if ref_doc_id == "img_doc":
        assert index.image_vector_store.to_dict()["embedding_dict"] == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_delete_nodes_cleans_text_and_image_vector_stores(
    use_async: bool,
) -> None:
    index = _build_index()
    img_node_id = next(iter(index.image_vector_store.to_dict()["embedding_dict"]))

    if use_async:
        await index.adelete_nodes([img_node_id], delete_from_docstore=True)
    else:
        index.delete_nodes([img_node_id], delete_from_docstore=True)

    assert index.image_vector_store.to_dict()["embedding_dict"] == {}
    assert img_node_id not in index.index_struct.nodes_dict
    assert "img_doc" not in index.ref_doc_info
