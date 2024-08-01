from typing import List
from llama_index.core.schema import Document
from llama_index.core.indices.managed.base import BaseManagedIndex
from llama_index.indices.managed.vectara import VectaraIndex
import pytest

#
# For this test to run properly, please setup as follows:
# 1. Create a Vectara account: sign up at https://console.vectara.com/signup
# 2. Create a corpus in your Vectara account, with a "filter attribute" called "test_num".
# 3. Create an API_KEY for this corpus with permissions for query and indexing
# 4. Setup environment variables:
#    VECTARA_API_KEY, VECTARA_CORPUS_ID and VECTARA_CUSTOMER_ID
#


def test_class():
    names_of_base_classes = [b.__name__ for b in VectaraIndex.__mro__]
    assert BaseManagedIndex.__name__ in names_of_base_classes


def get_docs() -> List[Document]:
    inputs = [
        {
            "text": "This is test text for Vectara integration with LlamaIndex",
            "metadata": {"test_num": "1"},
        },
        {
            "text": "And now for something completely different",
            "metadata": {"test_num": "2"},
        },
        {
            "text": "when 900 years you will be, look as good you will not",
            "metadata": {"test_num": "3"},
        },
        {
            "text": "when 850 years you will be, look as good you will not",
            "metadata": {"test_num": "4"},
        },
    ]
    docs: List[Document] = []
    for inp in inputs:
        doc = Document(
            text=str(inp["text"]),
            metadata=inp["metadata"],  # type: ignore
        )
        docs.append(doc)
    return docs


@pytest.fixture()
def vectara1():
    docs = get_docs()
    try:
        vectara1 = VectaraIndex.from_documents(docs)
    except ValueError:
        pytest.skip("Missing Vectara credentials, skipping test")

    yield vectara1

    # Tear down code
    for id in vectara1.doc_ids:
        vectara1._delete_doc(id)


def test_simple_retrieval(vectara1) -> None:
    docs = get_docs()
    qe = vectara1.as_retriever(similarity_top_k=1)
    res = qe.retrieve("how will I look?")
    assert len(res) == 1
    assert res[0].node.get_content() == docs[2].text


def test_mmr_retrieval(vectara1) -> None:
    docs = get_docs()

    # test with diversity bias = 0
    qe = vectara1.as_retriever(
        similarity_top_k=2,
        n_sentences_before=0,
        n_sentences_after=0,
        reranker="mmr",
        mmr_k=10,
        mmr_diversity_bias=0.0,
    )
    res = qe.retrieve("how will I look?")
    assert len(res) == 2
    assert res[0].node.get_content() == docs[2].text
    assert res[1].node.get_content() == docs[3].text

    # test with diversity bias = 1
    qe = vectara1.as_retriever(
        similarity_top_k=2,
        n_sentences_before=0,
        n_sentences_after=0,
        reranker="mmr",
        mmr_k=10,
        mmr_diversity_bias=1.0,
    )
    res = qe.retrieve("how will I look?")
    assert len(res) == 2
    assert res[0].node.get_content() == docs[2].text
    assert res[1].node.get_content() == docs[0].text


def test_retrieval_with_filter(vectara1) -> None:
    docs = get_docs()

    assert isinstance(vectara1, VectaraIndex)
    qe = vectara1.as_retriever(similarity_top_k=1, filter="doc.test_num = '1'")
    res = qe.retrieve("how will I look?")
    assert len(res) == 1
    assert res[0].node.get_content() == docs[0].text


@pytest.fixture()
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
    vectara2._delete_doc(id)


def test_file_upload(vectara2) -> None:
    # test query with Vectara summarization (default)
    query_engine = vectara2.as_query_engine(similarity_top_k=3)
    res = query_engine.query("What software did Paul Graham write?")
    assert "paul graham" in str(res).lower() and "software" in str(res).lower()
    assert "fcs" in res.metadata
    assert res.metadata["fcs"] >= 0

    # test query with Vectara summarization (streaming)
    query_engine = vectara2.as_query_engine(similarity_top_k=3, streaming=True)
    res = query_engine.query("What software did Paul Graham write?")
    summary = ""
    for chunk in res.response_gen:
        if chunk.delta:
            summary += chunk.delta
        if (
            chunk.additional_kwargs
            and "fcs" in chunk.additional_kwargs
            and chunk.additional_kwargs["fcs"] is not None
        ):
            assert chunk.additional_kwargs["fcs"] >= 0
    assert "paul graham" in summary.lower() and "software" in summary.lower()

    # test query with VectorStoreQuery (using OpenAI for summarization)
    query_engine = vectara2.as_query_engine(similarity_top_k=3, summary_enabled=False)
    res = query_engine.query("What software did Paul Graham write?")
    assert "paul graham" in str(res).lower() and "software" in str(res).lower()

    # test query with Vectara summarization (default)
    query_engine = vectara2.as_query_engine(
        similarity_top_k=3, citations_url_pattern="{doc.url}"
    )
    res = query_engine.query("How is Paul related to Reddit?")
    assert "paul graham" in str(res).lower() and "reddit" in str(res).lower()
    assert "https://www.paulgraham.com/worked.html" in str(res).lower()
