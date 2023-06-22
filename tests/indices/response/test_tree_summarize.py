"""Test tree summarize."""

from typing import List, Sequence
from unittest.mock import Mock

from llama_index.indices.prompt_helper import PromptHelper
from llama_index.indices.response.tree_summarize import TreeSummarize
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts.base import Prompt
from llama_index.prompts.prompt_type import PromptType


def test_tree_summarize(mock_service_context: ServiceContext) -> None:
    mock_qa_prompt_tmpl = "{context_str}{query_str}"
    mock_qa_prompt = Prompt(mock_qa_prompt_tmpl, prompt_type=PromptType.QUESTION_ANSWER)

    def mock_repack(prompt_template: Prompt, text_chunks: Sequence[str]) -> List[str]:
        merged_chunks = []
        for chunks in zip(*[iter(text_chunks)] * 2):
            merged_chunks.append("\n".join(chunks))
        return merged_chunks

    mock_prompt_helper = Mock(spec=PromptHelper)
    mock_prompt_helper.repack.side_effect = mock_repack
    mock_service_context.prompt_helper = mock_prompt_helper

    tree_summarize = TreeSummarize(
        service_context=mock_service_context,
        text_qa_template=mock_qa_prompt,
        verbose=True,
    )

    query_str = "What is?"
    texts = [
        "Text chunk 1",
        "Text chunk 2",
        "Text chunk 3",
        "Text chunk 4",
    ]

    response = tree_summarize.get_response(text_chunks=texts, query_str=query_str)
    assert str(response) == "Text chunk 1\nText chunk 2\nText chunk 3\nText chunk 4"
