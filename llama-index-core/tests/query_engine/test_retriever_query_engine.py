from llama_index.core import Document, TreeIndex
from llama_index.core.indices.tree.select_leaf_retriever import TreeSelectLeafRetriever
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
