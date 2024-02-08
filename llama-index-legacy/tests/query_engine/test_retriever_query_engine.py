import pytest
from llama_index.legacy import (
    Document,
    ServiceContext,
    TreeIndex,
)
from llama_index.legacy.indices.tree.select_leaf_retriever import (
    TreeSelectLeafRetriever,
)
from llama_index.legacy.llms import Anthropic
from llama_index.legacy.llms.openai import OpenAI
from llama_index.legacy.query_engine.retriever_query_engine import (
    RetrieverQueryEngine,
)

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore


@pytest.mark.skipif(anthropic is None, reason="anthropic not installed")
def test_query_engine_falls_back_to_inheriting_retrievers_service_context() -> None:
    documents = [Document(text="Hi")]
    gpt35turbo_predictor = OpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo-0613",
        streaming=True,
        openai_api_key="test-test-test",
    )
    gpt35_sc = ServiceContext.from_defaults(
        llm=gpt35turbo_predictor,
        chunk_size=512,
    )

    gpt35_tree_index = TreeIndex.from_documents(documents, service_context=gpt35_sc)
    retriever = TreeSelectLeafRetriever(index=gpt35_tree_index, child_branch_factor=2)
    query_engine = RetrieverQueryEngine(retriever=retriever)

    assert (
        retriever._service_context.llm.metadata.model_name
        == gpt35turbo_predictor.metadata.model_name
    )
    assert (
        query_engine._response_synthesizer.service_context.llm.metadata.model_name
        == retriever._service_context.llm.metadata.model_name
    )
    assert (
        query_engine._response_synthesizer.service_context == retriever._service_context
    )

    documents = [Document(text="Hi")]
    claude_predictor = Anthropic(model="claude-2")
    claude_sc = ServiceContext.from_defaults(
        llm=claude_predictor,
        chunk_size=512,
    )

    claude_tree_index = TreeIndex.from_documents(documents, service_context=claude_sc)
    retriever = TreeSelectLeafRetriever(index=claude_tree_index, child_branch_factor=2)
    query_engine = RetrieverQueryEngine(retriever=retriever)

    assert (
        retriever._service_context.llm.metadata.model_name
        == claude_predictor.metadata.model_name
    )
    assert (
        query_engine._response_synthesizer.service_context.llm.metadata.model_name
        == retriever._service_context.llm.metadata.model_name
    )
    assert (
        query_engine._response_synthesizer.service_context == retriever._service_context
    )
