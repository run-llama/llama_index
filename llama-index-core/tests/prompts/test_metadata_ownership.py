"""
A prompt constructor must not write into the metadata dict it was given.

All three template classes that accept ``metadata`` add a ``prompt_type`` entry to
it before handing it to ``BasePromptTemplate``, which means the dict the caller
owned comes back carrying a key the caller never set.
"""

from typing import Any, Dict, List

import pytest
from llama_index.core import ChatPromptTemplate, PromptTemplate
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts import PromptType


def _chat_messages() -> List[ChatMessage]:
    return [ChatMessage(role=MessageRole.USER, content="answer {question}")]


def test_prompt_template_leaves_the_callers_dict_alone() -> None:
    metadata: Dict[str, Any] = {"team": "rag"}

    prompt = PromptTemplate("answer {question}", metadata=metadata)

    assert metadata == {"team": "rag"}
    assert prompt.metadata["prompt_type"] == PromptType.CUSTOM
    assert prompt.metadata is not metadata


def test_chat_prompt_template_leaves_the_callers_dict_alone() -> None:
    metadata: Dict[str, Any] = {"team": "rag"}

    prompt = ChatPromptTemplate(_chat_messages(), metadata=metadata)

    assert metadata == {"team": "rag"}
    assert prompt.metadata["prompt_type"] == PromptType.CUSTOM


def test_one_dict_reused_by_several_prompts_stays_clean() -> None:
    metadata: Dict[str, Any] = {"team": "rag"}

    first = PromptTemplate("one {x}", metadata=metadata)
    second = PromptTemplate("two {x}", metadata=metadata, prompt_type="custom_two")

    assert metadata == {"team": "rag"}
    assert first.metadata["prompt_type"] == PromptType.CUSTOM
    assert second.metadata["prompt_type"] == "custom_two"


def test_metadata_supplied_by_the_caller_survives_untouched() -> None:
    metadata: Dict[str, Any] = {"prompt_type": "caller_value", "extra": 1}

    prompt = PromptTemplate("answer {question}", metadata=metadata)

    assert metadata == {"prompt_type": "caller_value", "extra": 1}
    assert prompt.metadata["prompt_type"] == PromptType.CUSTOM


def test_default_metadata_is_still_created() -> None:
    """The no-argument path keeps working, so the fix is not a behaviour change."""
    assert PromptTemplate("answer {question}").metadata["prompt_type"] == (
        PromptType.CUSTOM
    )


def test_langchain_prompt_template_leaves_the_callers_dict_alone() -> None:
    pytest.importorskip("langchain_core")
    from langchain_core.prompts import PromptTemplate as LangchainTemplate

    from llama_index.core.prompts import LangchainPromptTemplate

    metadata: Dict[str, Any] = {"team": "rag"}

    prompt = LangchainPromptTemplate(
        selector=LangchainTemplate.from_template("answer {question}"),
        metadata=metadata,
        prompt_type="langchain",
    )

    assert metadata == {"team": "rag"}
    assert prompt.metadata["prompt_type"] == "langchain"
