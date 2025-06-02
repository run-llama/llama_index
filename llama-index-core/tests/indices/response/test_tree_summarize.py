"""Test tree summarize."""

from typing import Any, List, Sequence, Optional
from unittest.mock import Mock, patch

import pytest
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.llms.mock import MockLLM
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.response_synthesizers import TreeSummarize


@pytest.fixture()
def mock_prompt_helper(patch_llm_predictor, patch_token_text_splitter):
    def mock_repack(
        prompt_template: PromptTemplate,
        text_chunks: Sequence[str],
        llm: Optional[Any] = None,
        tools: Optional[Any] = None,
    ) -> List[str]:
        merged_chunks = []
        for chunks in zip(*[iter(text_chunks)] * 2):
            merged_chunks.append("\n".join(chunks))
        return merged_chunks

    mock_prompt_helper = Mock(spec=PromptHelper)
    mock_prompt_helper.repack.side_effect = mock_repack
    return mock_prompt_helper


def test_tree_summarize(mock_prompt_helper) -> None:
    mock_summary_prompt_tmpl = "{context_str}{query_str}"
    mock_summary_prompt = PromptTemplate(
        mock_summary_prompt_tmpl, prompt_type=PromptType.SUMMARY
    )

    query_str = "What is?"
    texts = [
        "Text chunk 1",
        "Text chunk 2",
        "Text chunk 3",
        "Text chunk 4",
    ]

    # test sync
    tree_summarize = TreeSummarize(
        prompt_helper=mock_prompt_helper,
        summary_template=mock_summary_prompt,
    )
    response = tree_summarize.get_response(text_chunks=texts, query_str=query_str)
    assert str(response) == "Text chunk 1\nText chunk 2\nText chunk 3\nText chunk 4"


class TestModel(BaseModel):
    hello: str


def mock_return_class(*args: Any, **kwargs: Any) -> TestModel:
    return TestModel(hello="Test Chunk 5")


@patch.object(MockLLM, "structured_predict", mock_return_class)
def test_tree_summarize_output_cls(mock_prompt_helper) -> None:
    mock_summary_prompt_tmpl = "{context_str}{query_str}"
    mock_summary_prompt = PromptTemplate(
        mock_summary_prompt_tmpl, prompt_type=PromptType.SUMMARY
    )

    query_str = "What is?"
    texts = [
        '{"hello":"Test Chunk 1"}',
        '{"hello":"Test Chunk 2"}',
        '{"hello":"Test Chunk 3"}',
        '{"hello":"Test Chunk 4"}',
    ]
    response_dict = {"hello": "Test Chunk 5"}

    # test sync
    tree_summarize = TreeSummarize(
        prompt_helper=mock_prompt_helper,
        summary_template=mock_summary_prompt,
        output_cls=TestModel,
    )
    full_response = "\n".join(texts)
    response = tree_summarize.get_response(text_chunks=texts, query_str=query_str)
    assert isinstance(response, TestModel)
    assert response.model_dump() == response_dict


def test_tree_summarize_use_async(mock_prompt_helper) -> None:
    mock_summary_prompt_tmpl = "{context_str}{query_str}"
    mock_summary_prompt = PromptTemplate(
        mock_summary_prompt_tmpl, prompt_type=PromptType.SUMMARY
    )

    query_str = "What is?"
    texts = [
        "Text chunk 1",
        "Text chunk 2",
        "Text chunk 3",
        "Text chunk 4",
    ]

    # test async
    tree_summarize = TreeSummarize(
        prompt_helper=mock_prompt_helper,
        summary_template=mock_summary_prompt,
        use_async=True,
    )
    response = tree_summarize.get_response(text_chunks=texts, query_str=query_str)
    assert str(response) == "Text chunk 1\nText chunk 2\nText chunk 3\nText chunk 4"


@pytest.mark.asyncio
async def test_tree_summarize_async(mock_prompt_helper) -> None:
    mock_summary_prompt_tmpl = "{context_str}{query_str}"
    mock_summary_prompt = PromptTemplate(
        mock_summary_prompt_tmpl, prompt_type=PromptType.SUMMARY
    )

    query_str = "What is?"
    texts = [
        "Text chunk 1",
        "Text chunk 2",
        "Text chunk 3",
        "Text chunk 4",
    ]

    # test async
    tree_summarize = TreeSummarize(
        prompt_helper=mock_prompt_helper,
        summary_template=mock_summary_prompt,
    )
    response = await tree_summarize.aget_response(
        text_chunks=texts, query_str=query_str
    )
    assert str(response) == "Text chunk 1\nText chunk 2\nText chunk 3\nText chunk 4"
