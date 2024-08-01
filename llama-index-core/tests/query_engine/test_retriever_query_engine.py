import pytest
from llama_index.core import (
    Document,
    ServiceContext,
    TreeIndex,
)
from llama_index.core.indices.tree.select_leaf_retriever import (
    TreeSelectLeafRetriever,
)
from llama_index.core.query_engine.retriever_query_engine import (
    RetrieverQueryEngine,
)

try:
    from llama_index.llms.openai import OpenAI  # pants: no-infer-dep
except ImportError:
    OpenAI = None  # type: ignore


@pytest.mark.skipif(OpenAI is None, reason="llama-index-llms-openai not installed")
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
        retriever._llm.metadata.model_name == gpt35turbo_predictor.metadata.model_name
    )
    assert (
        query_engine._response_synthesizer._llm.metadata.model_name
        == retriever._llm.metadata.model_name
    )
    assert (
        query_engine._response_synthesizer.callback_manager
        == retriever.callback_manager
    )
