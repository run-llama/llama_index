from llama_index import (
    ServiceContext,
    LLMPredictor,
    TreeIndex,
    Document,
)
from llama_index.llms import Anthropic
from langchain.chat_models import ChatOpenAI
from llama_index.indices.tree.select_leaf_retriever import TreeSelectLeafRetriever
from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine


# We test two different models in case one ends up becoming the default of
# llama_index.llms.utils.resolve_llm
def test_query_engine_falls_back_to_inheriting_retrievers_service_context() -> None:
    documents = [Document(text="Hi")]
    gpt35turbo_predictor = LLMPredictor(
        llm=ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo-0613",
            streaming=True,
            openai_api_key="test-test-test",
        ),
    )
    gpt35_sc = ServiceContext.from_defaults(
        llm_predictor=gpt35turbo_predictor,
        chunk_size=512,
    )

    gpt35_tree_index = TreeIndex.from_documents(documents, service_context=gpt35_sc)
    retriever = TreeSelectLeafRetriever(index=gpt35_tree_index, child_branch_factor=2)
    query_engine = RetrieverQueryEngine(retriever=retriever)

    assert (
        retriever._service_context.llm_predictor.metadata.model_name
        == gpt35turbo_predictor._llm.metadata.model_name
    )
    assert (
        query_engine._response_synthesizer.service_context.llm_predictor.metadata.model_name
        == retriever._service_context.llm_predictor.metadata.model_name
    )
    assert (
        query_engine._response_synthesizer.service_context == retriever._service_context
    )

    documents = [Document(text="Hi")]
    claude_predictor = LLMPredictor(llm=Anthropic(model="claude-2"))
    claude_sc = ServiceContext.from_defaults(
        llm_predictor=claude_predictor,
        chunk_size=512,
    )

    claude_tree_index = TreeIndex.from_documents(documents, service_context=claude_sc)
    retriever = TreeSelectLeafRetriever(index=claude_tree_index, child_branch_factor=2)
    query_engine = RetrieverQueryEngine(retriever=retriever)

    assert (
        retriever._service_context.llm_predictor.metadata.model_name
        == claude_predictor._llm.metadata.model_name
    )
    assert (
        query_engine._response_synthesizer.service_context.llm_predictor.metadata.model_name
        == retriever._service_context.llm_predictor.metadata.model_name
    )
    assert (
        query_engine._response_synthesizer.service_context == retriever._service_context
    )
