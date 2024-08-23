from typing import Any, List
from unittest.mock import Mock, patch

from llama_index.core.chat_engine.condense_plus_context import (
    CondensePlusContextChatEngine,
)
from llama_index.core.indices.base_retriever import BaseRetriever
from llama_index.core.llms.mock import MockLLM
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.schema import NodeWithScore, TextNode


def override_predict(self: Any, prompt: BasePromptTemplate, **prompt_args: Any) -> str:
    return prompt.format(**prompt_args)


@patch.object(
    MockLLM,
    "predict",
    override_predict,
)
def test_condense_plus_context_chat_engine(mock_llm) -> None:
    mock_retriever = Mock(spec=BaseRetriever)

    def source_url(query: str) -> str:
        query_url = query.replace(" ", "_")
        # limit to first 10 characters
        query_url = query_url[:10]
        return f"http://example.com/{query_url}"

    def override_retrieve(query: str) -> List[NodeWithScore]:
        # replace spaces with underscore in query
        query_url = query.replace(" ", "_")
        return [
            NodeWithScore(
                node=TextNode(
                    text=query,
                    id_="id_100001",
                    metadata={
                        "source": source_url(query),
                    },
                ),
                score=0.9,
            )
        ]

    mock_retriever.retrieve.side_effect = override_retrieve

    context_prompt = "Context information: {context_str}"

    condense_prompt = (
        "Condense to a single question. Chat history: {chat_history}\n"
        "Follow up question: {question}\n"
        "Standalone question: "
    )

    engine = CondensePlusContextChatEngine(
        retriever=mock_retriever,
        llm=MockLLM(),
        memory=ChatMemoryBuffer.from_defaults(chat_history=[], llm=mock_llm),
        context_prompt=context_prompt,
        condense_prompt=condense_prompt,
    )

    engine.reset()
    input_1 = "First Query"
    actual_response_1 = engine.chat(input_1)

    # Keep reference of the mock source URL constructed for this input
    source_url_1 = source_url(input_1)
    # No condensing should happen for the first chat

    expected_response_str_1 = (
        f"system: Context information: source: {source_url_1}\n\n{input_1}"
        f"\nuser: {input_1}"
        f"\nassistant: "
    )
    assert str(actual_response_1) == expected_response_str_1
    # Check if the source nodes are correctly set
    assert actual_response_1.source_nodes == override_retrieve(input_1)

    input_2 = "Second Query"
    actual_response_2 = engine.chat(input_2)

    # For the second input, context will be fetched for the condensed query
    source_url_2 = source_url(condense_prompt)
    # Now condensing should happen for the previous chat history and new question
    expected_response_str_2 = (
        f"system: Context information: source: {source_url_2}\n\n"
        "Condense to a single question. Chat history: "
        f"user: {input_1}"
        f"\nassistant: {expected_response_str_1}"
        f"\nFollow up question: {input_2}"
        f"\nStandalone question:"
        f"\nuser: {input_1}"
        f"\nassistant: system: Context information: source: {source_url_1}\n\n{input_1}"
        f"\nuser: {input_1}"
        f"\nassistant: "
        f"\nuser: {input_2}"
        f"\nassistant: "
    )
    assert str(actual_response_2) == expected_response_str_2

    engine.reset()

    input_3 = "Fresh Query"
    actual_response_3 = engine.chat(input_3)

    # Keep reference of the mock source URL constructed for this input
    source_url_3 = source_url(input_3)
    # Now no condensing should happen as we did engine reset
    expected_response_str_3 = (
        f"system: Context information: source: {source_url_3}\n\n{input_3}"
        f"\nuser: {input_3}"
        f"\nassistant: "
    )
    assert str(actual_response_3) == expected_response_str_3
