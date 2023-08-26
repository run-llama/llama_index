"""Test tree summarize."""

from typing import List, Sequence
from unittest.mock import Mock

import pytest

from llama_index.indices.prompt_helper import PromptHelper
from llama_index.response_synthesizers import TreeSummarize
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.prompt_type import PromptType


@pytest.fixture()
def mock_service_context_merge_chunks(
    mock_service_context: ServiceContext,
) -> ServiceContext:
    def mock_repack(
        prompt_template: PromptTemplate, text_chunks: Sequence[str]
    ) -> List[str]:
        merged_chunks = []
        for chunks in zip(*[iter(text_chunks)] * 2):
            merged_chunks.append("\n".join(chunks))
        return merged_chunks

    mock_prompt_helper = Mock(spec=PromptHelper)
    mock_prompt_helper.repack.side_effect = mock_repack
    mock_service_context.prompt_helper = mock_prompt_helper
    return mock_service_context


def test_tree_summarize(mock_service_context_merge_chunks: ServiceContext) -> None:
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
        service_context=mock_service_context_merge_chunks,
        summary_template=mock_summary_prompt,
    )
    response = tree_summarize.get_response(text_chunks=texts, query_str=query_str)
    assert str(response) == "Text chunk 1\nText chunk 2\nText chunk 3\nText chunk 4"


def test_tree_summarize_use_async(
    mock_service_context_merge_chunks: ServiceContext,
) -> None:
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
        service_context=mock_service_context_merge_chunks,
        summary_template=mock_summary_prompt,
        use_async=True,
    )
    response = tree_summarize.get_response(text_chunks=texts, query_str=query_str)
    assert str(response) == "Text chunk 1\nText chunk 2\nText chunk 3\nText chunk 4"


@pytest.mark.asyncio
async def test_tree_summarize_async(
    mock_service_context_merge_chunks: ServiceContext,
) -> None:
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
        service_context=mock_service_context_merge_chunks,
        summary_template=mock_summary_prompt,
    )
    response = await tree_summarize.aget_response(
        text_chunks=texts, query_str=query_str
    )
    assert str(response) == "Text chunk 1\nText chunk 2\nText chunk 3\nText chunk 4"
