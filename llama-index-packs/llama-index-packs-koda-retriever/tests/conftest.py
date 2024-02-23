from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.postprocessor import LLMRerank
from llama_index.core import Settings
from llama_index.packs.koda_retriever import (
    KodaRetriever,
    AlphaMatrix,
    DEFAULT_CATEGORIES,
)
import pytest

from tests.koda_mocking import KVMockLLM
from tests.monkeypatch import monkey_patch_vector_store_index


@pytest.fixture()
def setup() -> dict:  # monkey
    """Sets up fixtures for simple vector stores to be used within retriever testing."""
    Settings.llm = KVMockLLM()
    Settings.embed_model = MockEmbedding(8)

    # add text nodes custom
    vector_index = monkey_patch_vector_store_index()

    shots = AlphaMatrix(
        data=DEFAULT_CATEGORIES
    )  # this could also just be a dictionary i guess

    reranker = LLMRerank(llm=Settings.llm)

    retriever = KodaRetriever(
        index=vector_index,
        llm=Settings.llm,
        reranker=reranker,
        matrix=shots,
        verbose=True,
    )

    return {
        "retriever": retriever,
        "llm": Settings.llm,
        "reranker": reranker,
        "embed_model": Settings.embed_model,
        "vector_index": vector_index,
        "matrix": shots,
    }
