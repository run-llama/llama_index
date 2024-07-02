from llama_index.embeddings.litellm import LiteLLMEmbedding

MOCK_EMBEDDINGS = [[0.1, 0.2, 0.3]]


def test_get_query_embedding(mocker):
    mocker.patch(
        "llama_index.embeddings.litellm.base.get_embeddings",
        return_value=MOCK_EMBEDDINGS,
    )
    lite_llm_embedding = LiteLLMEmbedding(api_base="test_base", model_name="test_model")

    result = lite_llm_embedding._get_query_embedding("test_query")
    assert result == [0.1, 0.2, 0.3]


def test_get_text_embedding(mocker):
    mocker.patch(
        "llama_index.embeddings.litellm.base.get_embeddings",
        return_value=MOCK_EMBEDDINGS,
    )
    lite_llm_embedding = LiteLLMEmbedding(api_base="test_base", model_name="test_model")

    result = lite_llm_embedding._get_text_embedding("test_query")
    assert result == [0.1, 0.2, 0.3]


def test_get_text_embeddings(mocker):
    mocker.patch(
        "llama_index.embeddings.litellm.base.get_embeddings",
        return_value=MOCK_EMBEDDINGS,
    )
    lite_llm_embedding = LiteLLMEmbedding(api_base="test_base", model_name="test_model")

    result = lite_llm_embedding._get_text_embeddings("test_query")
    assert result == [[0.1, 0.2, 0.3]]
