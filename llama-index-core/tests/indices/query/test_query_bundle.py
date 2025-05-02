"""Test query bundle."""

from typing import Dict, List

import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.indices.list.base import SummaryIndex
from llama_index.core.schema import Document, QueryBundle


@pytest.fixture()
def documents() -> List[Document]:
    """Get documents."""
    # NOTE: one document for now
    doc_text = (
        "Correct.\n"
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    return [Document(text=doc_text)]


class MockEmbedding(BaseEmbedding):
    @classmethod
    def class_name(cls) -> str:
        return "MockEmbedding"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        text_embed_map: Dict[str, List[float]] = {
            "It is what it is.": [1.0, 0.0, 0.0, 0.0, 0.0],
            "The meaning of life": [0.0, 1.0, 0.0, 0.0, 0.0],
        }

        return text_embed_map[query]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        text_embed_map: Dict[str, List[float]] = {
            "Correct.": [0.5, 0.5, 0.0, 0.0, 0.0],
            "Hello world.": [1.0, 0.0, 0.0, 0.0, 0.0],
            "This is a test.": [0.0, 1.0, 0.0, 0.0, 0.0],
            "This is another test.": [0.0, 0.0, 1.0, 0.0, 0.0],
            "This is a test v2.": [0.0, 0.0, 0.0, 1.0, 0.0],
        }

        return text_embed_map[text]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get node text embedding."""
        text_embed_map: Dict[str, List[float]] = {
            "Correct.": [0.5, 0.5, 0.0, 0.0, 0.0],
            "Hello world.": [1.0, 0.0, 0.0, 0.0, 0.0],
            "This is a test.": [0.0, 1.0, 0.0, 0.0, 0.0],
            "This is another test.": [0.0, 0.0, 1.0, 0.0, 0.0],
            "This is a test v2.": [0.0, 0.0, 0.0, 1.0, 0.0],
        }

        return text_embed_map[text]

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        text_embed_map: Dict[str, List[float]] = {
            "It is what it is.": [1.0, 0.0, 0.0, 0.0, 0.0],
            "The meaning of life": [0.0, 1.0, 0.0, 0.0, 0.0],
        }

        return text_embed_map[query]


def test_embedding_query(
    documents: List[Document], patch_llm_predictor, patch_token_text_splitter
) -> None:
    """Test embedding query."""
    index = SummaryIndex.from_documents(documents)

    # test embedding query
    query_bundle = QueryBundle(
        query_str="What is?",
        custom_embedding_strs=[
            "It is what it is.",
            "The meaning of life",
        ],
    )
    retriever = index.as_retriever(
        retriever_mode="embedding", similarity_top_k=1, embed_model=MockEmbedding()
    )
    nodes = retriever.retrieve(query_bundle)
    assert len(nodes) == 1
    assert nodes[0].node.get_content() == "Correct."
