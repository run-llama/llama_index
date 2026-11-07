"""Regression tests for MultiModalVectorStoreIndex.delete_ref_doc."""

import base64
import os

import pytest

from llama_index.core.embeddings import MockMultiModalEmbedding
from llama_index.core.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.core.schema import ImageDocument

_PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture()
def image_doc(tmp_path):
    image_path = os.path.join(tmp_path, "tiny.png")
    with open(image_path, "wb") as f:
        f.write(_PNG_1PX)
    return ImageDocument(text="a note", image_path=image_path)


def test_delete_ref_doc_with_store_nodes_override(image_doc) -> None:
    """Deleting a stored-node multimodal doc must not raise KeyError."""
    index = MultiModalVectorStoreIndex.from_documents(
        [image_doc],
        embed_model=MockMultiModalEmbedding(embed_dim=4),
        image_embed_model=MockMultiModalEmbedding(embed_dim=4),
        store_nodes_override=True,
        show_progress=False,
    )
    # A multimodal index always has at least two vector stores.
    assert len(index._storage_context.vector_stores) >= 2

    index.delete_ref_doc(image_doc.get_doc_id(), delete_from_docstore=True)

    assert index._docstore.get_ref_doc_info(image_doc.get_doc_id()) is None
    assert index.ref_doc_info.get(image_doc.get_doc_id()) is None
