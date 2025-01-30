from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference


def test_text_inference_embedding_class():
    names_of_base_classes = [b.__name__ for b in TextEmbeddingsInference.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes


def test_text_inference_embedding_init():
    text_inference = TextEmbeddingsInference(
        model_name="some-model",
        base_url="some-url",
        text_instruction="some-text-instruction",
        query_instruction="some-query-instruction",
        embed_batch_size=42,
        timeout=42.0,
        truncate_text=False,
        auth_token="some-token",
        endpoint="some-endpoint",
    )
    assert text_inference.model_name == "some-model"
    assert text_inference.base_url == "some-url"
    assert text_inference.text_instruction == "some-text-instruction"
    assert text_inference.query_instruction == "some-query-instruction"
    assert text_inference.embed_batch_size == 42
    assert int(text_inference.timeout) == 42
    assert text_inference.truncate_text is False
    assert text_inference.auth_token == "some-token"
    assert text_inference.endpoint == "some-endpoint"
