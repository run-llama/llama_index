from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.orcarouter import OrcaRouter
from llama_index.llms.orcarouter.base import (
    ATTRIBUTION_REFERER,
    ATTRIBUTION_TITLE,
    DEFAULT_API_BASE,
    DEFAULT_MODEL,
)


def test_class_inheritance() -> None:
    names_of_base_classes = [b.__name__ for b in OrcaRouter.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_defaults() -> None:
    llm = OrcaRouter(api_key="dummy_key")
    assert llm.model == DEFAULT_MODEL
    assert llm.api_base == DEFAULT_API_BASE
    assert llm.is_chat_model is True


def test_attribution_headers_default() -> None:
    llm = OrcaRouter(api_key="dummy_key")
    assert llm.default_headers["HTTP-Referer"] == ATTRIBUTION_REFERER
    assert llm.default_headers["X-Title"] == ATTRIBUTION_TITLE


def test_attribution_headers_user_override() -> None:
    llm = OrcaRouter(
        api_key="dummy_key",
        default_headers={"X-Title": "my-app", "X-Custom": "v1"},
    )
    # User-supplied keys take precedence.
    assert llm.default_headers["X-Title"] == "my-app"
    assert llm.default_headers["X-Custom"] == "v1"
    # Unspecified attribution headers are still populated.
    assert llm.default_headers["HTTP-Referer"] == ATTRIBUTION_REFERER


def test_fallback_models_injected() -> None:
    llm = OrcaRouter(
        api_key="dummy_key",
        model="openai/gpt-4o-mini",
        fallback_models=["openai/gpt-4o", "anthropic/claude-sonnet-4.6"],
    )

    extra_body = llm.additional_kwargs.get("extra_body")
    assert isinstance(extra_body, dict)
    assert extra_body["models"] == [
        "openai/gpt-4o",
        "anthropic/claude-sonnet-4.6",
    ]
    assert extra_body["route"] == "fallback"


def test_fallback_models_preserves_user_extra_body() -> None:
    llm = OrcaRouter(
        api_key="dummy_key",
        model="openai/gpt-4o-mini",
        fallback_models=["openai/gpt-4o"],
        additional_kwargs={
            "extra_body": {"route": "custom", "models": ["already/set"]}
        },
    )
    extra_body = llm.additional_kwargs["extra_body"]
    # User-supplied extra_body values win.
    assert extra_body["route"] == "custom"
    assert extra_body["models"] == ["already/set"]
