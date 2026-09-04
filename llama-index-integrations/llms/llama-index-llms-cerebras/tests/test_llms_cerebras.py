from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.cerebras import Cerebras


def test_text_inference_embedding_class():
    names_of_base_classes = [b.__name__ for b in Cerebras.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_max_completion_tokens_is_preserved_without_alias():
    llm = Cerebras(
        model="gpt-oss-120b",
        api_key="test-key",
        max_tokens=32,
        max_completion_tokens=64,
    )

    params = llm._get_model_kwargs()

    assert params["max_completion_tokens"] == 64
    assert "max_tokens" not in params


def test_reasoning_effort_is_forwarded_for_cerebras_models():
    llm = Cerebras(
        model="gemma-4-31b",
        api_key="test-key",
        reasoning_effort="none",
    )

    assert llm._get_model_kwargs()["reasoning_effort"] == "none"
