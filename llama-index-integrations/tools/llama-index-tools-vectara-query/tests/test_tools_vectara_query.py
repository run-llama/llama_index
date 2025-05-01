import re
from typing import List

import pytest
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.schema import Document, MediaResource, Node
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.indices.managed.vectara import VectaraIndex
from llama_index.tools.vectara_query import VectaraQueryToolSpec

#
# For this test to run properly, please setup as follows:
# 1. Create a Vectara account: sign up at https://console.vectara.com/signup
# 2. Create two corpora in your Vectara account with the following filter attributes in the first corpus:
#   a. doc.test_num (text)
#   b. doc.test_score (integer)
#   c. doc.date (text)
#   d. doc.url (text)
# 3. Create an API_KEY for these corpora with permissions for query and indexing
# 4. Setup environment variables:
#    VECTARA_API_KEY, VECTARA_CORPUS_KEY, and OPENAI_API_KEY
#    For VECTARA_CORPUS_KEY, separate the corpus keys for the corpora with a ',' for example: "llamaindex-testing-1,llamaindex-testing-2".
#


def test_class():
    names_of_base_classes = [b.__name__ for b in VectaraQueryToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def get_docs() -> List[Document]:
    inputs = [
        {
            "id": "doc_1",
            "text": "This is test text for Vectara integration with LlamaIndex",
            "metadata": {"test_num": "1", "test_score": 10, "date": "2020-02-25"},
        },
        {
            "id": "doc_2",
            "text": "And now for something completely different",
            "metadata": {"test_num": "2", "test_score": 2, "date": "2015-10-13"},
        },
        {
            "id": "doc_3",
            "text": "when 900 years you will be, look as good you will not",
            "metadata": {"test_num": "3", "test_score": 20, "date": "2023-09-12"},
        },
        {
            "id": "doc_4",
            "text": "when 850 years you will be, look as good you will not",
            "metadata": {"test_num": "4", "test_score": 50, "date": "2022-01-01"},
        },
    ]
    docs: List[Document] = []
    for inp in inputs:
        doc = Document(
            id_=inp["id"],
            text_resource=MediaResource(text=inp["text"]),
            metadata=inp["metadata"],
        )
        docs.append(doc)
    return docs


def get_nodes() -> List[Node]:
    inputs = [
        {
            "text": "This is test text for Vectara integration with LlamaIndex",
            "metadata": {"test_num": "1", "test_score": 10, "date": "2020-02-25"},
        },
        {
            "text": "And now for something completely different",
            "metadata": {"test_num": "2", "test_score": 2, "date": "2015-10-13"},
        },
        {
            "text": "when 900 years you will be, look as good you will not",
            "metadata": {"test_num": "3", "test_score": 20, "date": "2023-09-12"},
        },
        {
            "text": "when 850 years you will be, look as good you will not",
            "metadata": {"test_num": "4", "test_score": 50, "date": "2022-01-01"},
        },
    ]

    nodes: List[Node] = []
    for inp in inputs:
        node = Node(
            text_resource=MediaResource(text=inp["text"]), metadata=inp["metadata"]
        )
        nodes.append(node)
    return nodes


@pytest.fixture
def vectara1():
    docs = get_docs()
    try:
        vectara1 = VectaraIndex.from_documents(docs)
    except ValueError:
        pytest.skip("Missing Vectara credentials, skipping test")

    yield vectara1

    # Tear down code
    for id in vectara1.doc_ids:
        vectara1.delete_ref_doc(id)


def test_simple_retrieval(vectara1) -> None:
    docs = get_docs()
    tool_spec = VectaraQueryToolSpec(num_results=1)
    res = tool_spec.semantic_search("Find me something different.")
    assert len(res) == 1
    assert res[0]["text"] == docs[1].text_resource.text
    assert (
        res[0]["citation_metadata"]["document"]["test_score"]
        == docs[1].metadata["test_score"]
    )


def test_mmr_retrieval(vectara1) -> None:
    docs = get_docs()

    # test with diversity bias = 0
    tool_spec = VectaraQueryToolSpec(
        num_results=2,
        n_sentences_before=0,
        n_sentences_after=0,
        reranker="mmr",
        rerank_k=10,
        mmr_diversity_bias=0.0,
    )
    res = tool_spec.semantic_search("How will I look?")
    assert len(res) == 2
    assert res[0]["text"] == docs[2].text_resource.text
    assert res[1]["text"] == docs[3].text_resource.text

    # test with diversity bias = 1
    tool_spec = VectaraQueryToolSpec(
        num_results=2,
        n_sentences_before=0,
        n_sentences_after=0,
        reranker="mmr",
        rerank_k=10,
        mmr_diversity_bias=1.0,
    )
    res = tool_spec.semantic_search("How will I look?")
    assert len(res) == 2
    assert res[0]["text"] == docs[2].text_resource.text


def test_retrieval_with_filter(vectara1) -> None:
    docs = get_docs()

    tool_spec = VectaraQueryToolSpec(
        num_results=1, metadata_filter=["doc.test_num = '1'", ""]
    )
    res = tool_spec.semantic_search("What does this test?")
    assert len(res) == 1
    assert res[0]["text"] == docs[0].text_resource.text


def test_udf_retrieval(vectara1) -> None:
    docs = get_docs()

    # test with basic math expression
    tool_spec = VectaraQueryToolSpec(
        num_results=2,
        n_sentences_before=0,
        n_sentences_after=0,
        reranker="userfn",
        udf_expression="get('$.score') + get('$.document_metadata.test_score')",
    )

    res = tool_spec.semantic_search("What will the future look like?")
    assert len(res) == 2
    assert res[0]["text"] == docs[3].text_resource.text
    assert res[1]["text"] == docs[2].text_resource.text

    # test with dates: Weight of score subtracted by number of years from current date
    tool_spec = VectaraQueryToolSpec(
        num_results=2,
        n_sentences_before=0,
        n_sentences_after=0,
        reranker="userfn",
        udf_expression="max(0, 5 * get('$.score') - (to_unix_timestamp(now()) - to_unix_timestamp(datetime_parse(get('$.document_metadata.date'), 'yyyy-MM-dd'))) / 31536000)",
    )

    res = tool_spec.semantic_search("What will the future look like?")
    assert len(res) == 2
    assert res[0]["text"] == docs[2].text_resource.text
    assert res[1]["text"] == docs[3].text_resource.text


def test_chain_rerank_retrieval(vectara1) -> None:
    docs = get_docs()

    # Test basic chain
    tool_spec = VectaraQueryToolSpec(
        num_results=2,
        n_sentences_before=0,
        n_sentences_after=0,
        reranker="chain",
        rerank_chain=[{"type": "slingshot"}, {"type": "mmr", "diversity_bias": 0.4}],
    )

    res = tool_spec.semantic_search("What's this all about?")
    assert len(res) == 2
    assert res[0]["text"] == docs[0].text_resource.text

    # Test chain with UDF and limit
    tool_spec = VectaraQueryToolSpec(
        num_results=4,
        n_sentences_before=0,
        n_sentences_after=0,
        reranker="chain",
        rerank_chain=[
            {"type": "slingshot"},
            {"type": "mmr"},
            {
                "type": "userfn",
                "user_function": "5 * get('$.score') + get('$.document_metadata.test_score') / 2",
                "limit": 2,
            },
        ],
    )

    res = tool_spec.semantic_search("What's this all about?")
    assert len(res) == 2
    assert res[0]["text"] == docs[3].text_resource.text
    assert res[1]["text"] == docs[2].text_resource.text

    # Test chain with cutoff
    tool_spec = VectaraQueryToolSpec(
        num_results=4,
        n_sentences_before=0,
        n_sentences_after=0,
        reranker="chain",
        rerank_chain=[
            {"type": "slingshot"},
            {"type": "mmr", "diversity_bias": 0.4, "cutoff": 0.75},
        ],
    )

    res = tool_spec.semantic_search("What's this all about?")
    assert len(res) == 1
    assert res[0]["text"] == docs[0].text_resource.text

    # Second query with same retriever to ensure rerank chain configuration remains the same
    res = tool_spec.semantic_search("How will I look when I'm older?")
    assert tool_spec.retriever._rerank_chain[0].get("type") == "customer_reranker"
    assert (
        tool_spec.retriever._rerank_chain[0].get("reranker_name")
        == "Rerank_Multilingual_v1"
    )
    assert tool_spec.retriever._rerank_chain[1].get("type") == "mmr"
    assert res[0]["text"] == docs[2].text_resource.text


def test_custom_prompt(vectara1) -> None:
    docs = get_docs()

    tool_spec = VectaraQueryToolSpec(
        num_results=3,
        n_sentences_before=0,
        n_sentences_after=0,
        reranker="mmr",
        mmr_diversity_bias=0.2,
        prompt_text='[\n  {"role": "system", "content": "You are an expert in summarizing the future of Vectara\'s inegration with LlamaIndex. Your summaries are insightful, concise, and highlight key innovations and changes."},\n  #foreach ($result in $vectaraQueryResults)\n    {"role": "user", "content": "What are the key points in result number $vectaraIdxWord[$foreach.index] about Vectara\'s LlamaIndex integration?"},\n    {"role": "assistant", "content": "In result number $vectaraIdxWord[$foreach.index], the key points are: ${result.getText()}"},\n  #end\n  {"role": "user", "content": "Can you generate a comprehensive summary on \'Vectara\'s LlamaIndex Integration\' incorporating all the key points discussed?"}\n]\n',
    )

    res = tool_spec.rag_query("How will Vectara's integration look in the future?")
    assert "integration" in res["summary"].lower()
    assert "llamaindex" in res["summary"].lower()
    assert "vectara" in res["summary"].lower()
    assert "result" in res["summary"].lower()
    assert res["factual_consistency_score"] > 0


@pytest.fixture
def vectara2():
    try:
        vectara2 = VectaraIndex()
    except ValueError:
        pytest.skip("Missing Vectara credentials, skipping test")

    file_path = "docs/docs/examples/data/paul_graham/paul_graham_essay.txt"
    id = vectara2.insert_file(
        file_path, metadata={"url": "https://www.paulgraham.com/worked.html"}
    )

    yield vectara2

    # Tear down code
    vectara2.delete_ref_doc(id)


def test_basic_rag_query(vectara2) -> None:
    # test query with Vectara summarization (default)
    tool_spec = VectaraQueryToolSpec(num_results=3)
    res = tool_spec.rag_query("What software did Paul Graham write?")
    assert (
        "paul graham" in res["summary"].lower() and "software" in res["summary"].lower()
    )
    assert "factual_consistency_score" in res
    assert res["factual_consistency_score"] >= 0

    res = tool_spec.rag_query("How is Paul related to Reddit?")
    summary = res["summary"]
    assert "paul graham" in summary.lower() and "reddit" in summary.lower()
    assert "https://www.paulgraham.com/worked.html" in str(res["citation_metadata"])


def test_citations(vectara2) -> None:
    # test markdown citations
    tool_spec = VectaraQueryToolSpec(
        num_results=10,
        summary_num_results=7,
        summarizer_prompt_name="vectara-summary-ext-24-05-med-omni",
        citations_style="markdown",
        citations_url_pattern="{doc.url}",
        citations_text_pattern="(source)",
    )
    res = tool_spec.rag_query("What colleges has Paul attended?")
    summary = res["summary"]
    assert "(source)" in summary
    assert "https://www.paulgraham.com/worked.html" in summary

    # test numeric citations
    tool_spec = VectaraQueryToolSpec(
        num_results=10,
        summary_num_results=7,
        summarizer_prompt_name="mockingbird-1.0-2024-07-16",
        citations_style="numeric",
    )
    res = tool_spec.rag_query("What colleges has Paul attended?")
    summary = res["summary"]
    assert re.search(r"\[\d+\]", summary)


@pytest.fixture
def vectara3():
    nodes = get_nodes()
    try:
        vectara3 = VectaraIndex()
        vectara3.add_nodes(
            nodes,
            document_id="doc_1",
            document_metadata={"author": "Vectara", "title": "LlamaIndex Integration"},
            corpus_key="llamaindex-testing-2",
        )
    except ValueError:
        pytest.skip("Missing Vectara credentials, skipping test")

    yield vectara3

    # Tear down code
    for id in vectara3.doc_ids:
        vectara3.delete_ref_doc(id, corpus_key="llamaindex-testing-2")


def test_metadata_format(vectara3) -> None:
    nodes = get_nodes()
    tool_spec = VectaraQueryToolSpec(
        num_results=1, n_sentences_before=0, n_sentences_after=0
    )
    res = tool_spec.semantic_search("Find me something different")
    assert len(res) == 1
    assert res[0]["citation_metadata"]["document"]["author"] == "Vectara"
    assert res[0]["citation_metadata"]["document"]["title"] == "LlamaIndex Integration"
    assert res[0]["text"] == nodes[1].text_resource.text
    assert res[0]["citation_metadata"]["test_score"] == nodes[1].metadata["test_score"]


def test_agent_basic(vectara2) -> None:
    tool_spec = VectaraQueryToolSpec(num_results=10, reranker="slingshot")
    agent = OpenAIAgent.from_tools(tool_spec.to_tool_list())
    res = agent.chat("What software did Paul Graham write?").response
    agent_tasks = agent.get_completed_tasks()
    tool_called = (
        agent_tasks[0]
        .memory.chat_store.store["chat_history"][1]
        .additional_kwargs["tool_calls"][0]
        .function.name
    )
    assert tool_called in ["semantic_search", "rag_query"]
    assert "paul graham" in res.lower() and "software" in res.lower()

    tool_spec = VectaraQueryToolSpec(num_results=10, reranker="mmr")
    agent = OpenAIAgent.from_tools(tool_spec.to_tool_list())
    res = agent.chat("Please summarize Paul's thoughts about paintings?").response
    agent_tasks = agent.get_completed_tasks()
    tool_called = (
        agent_tasks[0]
        .memory.chat_store.store["chat_history"][1]
        .additional_kwargs["tool_calls"][0]
        .function.name
    )
    assert tool_called == "rag_query"
    assert "paint" in res.lower() and "paul" in res.lower()


def test_agent_filter(vectara1) -> None:
    tool_spec = VectaraQueryToolSpec(
        num_results=1, metadata_filter=["doc.date > '2022-02-01'", ""]
    )

    agent = OpenAIAgent.from_tools(tool_spec.to_tool_list())

    res = agent.chat("How will I look when I am much older compared to now?").response
    agent_tasks = agent.get_completed_tasks()
    tool_called = (
        agent_tasks[0]
        .memory.chat_store.store["chat_history"][1]
        .additional_kwargs["tool_calls"][0]
        .function.name
    )
    assert tool_called in ["semantic_search", "rag_query"]
    assert "you" in res.lower()
