from unittest.mock import MagicMock, patch

from llama_index.core import Document, TreeIndex
from llama_index.core.indices.tree.select_leaf_retriever import TreeSelectLeafRetriever
from llama_index.core.llms.mock import MockLLM
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.prompts.chat_prompts import (
    CHAT_CONTENT_QA_PROMPT,
    CHAT_CONTENT_REFINE_PROMPT,
)
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core import Settings


def test_query_engine_falls_back_to_inheriting_retrievers_service_context(
    monkeypatch, mock_llm
) -> None:
    documents = [Document(text="Hi")]
    monkeypatch.setattr(Settings, "llm", mock_llm)

    gpt35_tree_index = TreeIndex.from_documents(documents)
    retriever = TreeSelectLeafRetriever(index=gpt35_tree_index, child_branch_factor=2)
    query_engine = RetrieverQueryEngine(retriever=retriever)

    assert retriever._llm.class_name() == "MockLLM"
    assert (
        query_engine._response_synthesizer._llm.metadata.model_name
        == retriever._llm.metadata.model_name
    )
    assert (
        query_engine._response_synthesizer.callback_manager
        == retriever.callback_manager
    )


def test_from_args_multimodal__builds_multimodal_synthesizer() -> None:
    retriever = MagicMock()
    with patch(
        "llama_index.core.response_synthesizers.base.is_chat_model",
        return_value=True,
    ):
        engine = RetrieverQueryEngine.from_args(
            retriever=retriever, llm=MockLLM(), multimodal=True
        )
    synth = engine._response_synthesizer
    assert synth._multimodal is True
    assert synth._chat_content_qa_template is CHAT_CONTENT_QA_PROMPT
    assert synth._chat_content_refine_template is CHAT_CONTENT_REFINE_PROMPT


def test_from_args_multimodal__overrides_chat_templates() -> None:
    retriever = MagicMock()
    custom_qa = RichPromptTemplate('{% chat role="user" %}QA{% endchat %}')
    custom_refine = RichPromptTemplate('{% chat role="user" %}REFINE{% endchat %}')
    with patch(
        "llama_index.core.response_synthesizers.base.is_chat_model",
        return_value=True,
    ):
        engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            llm=MockLLM(),
            multimodal=True,
            chat_content_qa_template=custom_qa,
            chat_content_refine_template=custom_refine,
        )
    synth = engine._response_synthesizer
    assert synth._chat_content_qa_template is custom_qa
    assert synth._chat_content_refine_template is custom_refine


def test_from_args_default__not_multimodal() -> None:
    engine = RetrieverQueryEngine.from_args(retriever=MagicMock(), llm=MockLLM())
    assert engine._response_synthesizer._multimodal is False
