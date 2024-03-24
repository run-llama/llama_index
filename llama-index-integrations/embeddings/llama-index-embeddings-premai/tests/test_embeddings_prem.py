from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.premai import PremAIEmbeddings


def test_embedding_class():
    emb = PremAIEmbeddings(
        project_id=8, model_name="text-embedding-3-large", premai_api_key="test"
    )
    assert isinstance(emb, BaseEmbedding)
