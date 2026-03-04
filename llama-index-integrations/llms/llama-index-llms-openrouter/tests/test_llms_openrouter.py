from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.openrouter import OpenRouter


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in OpenRouter.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_provider_routing_settings_injected() -> None:
    llm = OpenRouter(
        api_key="dummy_key",
        model="mistralai/mixtral-8x7b-instruct",
        order=["openai", "together"],
        allow_fallbacks=False,
        additional_kwargs={
            "extra_body": {
                "provider": {
                    "require_parameters": True,
                }
            }
        },
    )

    extra_body = llm.additional_kwargs.get("extra_body")
    assert isinstance(extra_body, dict)

    provider = extra_body.get("provider")
    assert isinstance(provider, dict)

    assert provider["order"] == ["openai", "together"]
    assert provider["allow_fallbacks"] is False
    assert provider["require_parameters"] is True
