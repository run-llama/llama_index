from typing import List

import pytest
from llama_index.legacy.indices.managed.vectara.base import VectaraIndex
from llama_index.legacy.schema import Document

#
# For this test to run properly, please setup as follows:
# 1. Create a Vectara account: sign up at https://console.vectara.com/signup
# 2. Create a corpus in your Vectara account, with a "filter attribute" called "test_num".
# 3. Create an API_KEY for this corpus with permissions for query and indexing
# 4. Setup environment variables:
#    VECTARA_API_KEY, VECTARA_CORPUS_ID and VECTARA_CUSTOMER_ID
#


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


def remove_docs(index: VectaraIndex, ids: List) -> None:
    for id in ids:
        index._delete_doc(id)


def test_simple_retrieval() -> None:
    docs = get_docs()
    try:
        index = VectaraIndex.from_documents(docs)
    except ValueError:
        pytest.skip("Missing Vectara credentials, skipping test")

    assert isinstance(index, VectaraIndex)
    qe = index.as_retriever(similarity_top_k=1)
    res = qe.retrieve("how will I look?")
    assert len(res) == 1
    assert res[0].node.get_content() == docs[2].text

    remove_docs(index, index.doc_ids)


def test_mmr_retrieval() -> None:
    docs = get_docs()
    try:
        index = VectaraIndex.from_documents(docs)
    except ValueError:
        pytest.skip("Missing Vectara credentials, skipping test")

    assert isinstance(index, VectaraIndex)

    # test with diversity bias = 0
    qe = index.as_retriever(
        similarity_top_k=2,
        n_sentences_before=0,
        n_sentences_after=0,
        vectara_query_mode="mmr",
        mmr_k=10,
        mmr_diversity_bias=0.0,
    )
    res = qe.retrieve("how will I look?")
    assert len(res) == 2
    assert res[0].node.get_content() == docs[2].text
    assert res[1].node.get_content() == docs[3].text

    # test with diversity bias = 1
    qe = index.as_retriever(
        similarity_top_k=2,
        n_sentences_before=0,
        n_sentences_after=0,
        vectara_query_mode="mmr",
        mmr_k=10,
        mmr_diversity_bias=1.0,
    )
    res = qe.retrieve("how will I look?")
    assert len(res) == 2
    assert res[0].node.get_content() == docs[2].text
    assert res[1].node.get_content() == docs[0].text

    remove_docs(index, index.doc_ids)


def test_retrieval_with_filter() -> None:
    docs = get_docs()
    try:
        index = VectaraIndex.from_documents(docs)
    except ValueError:
        pytest.skip("Missing Vectara credentials, skipping test")

    assert isinstance(index, VectaraIndex)
    qe = index.as_retriever(similarity_top_k=1, filter="doc.test_num = '1'")
    res = qe.retrieve("how will I look?")
    assert len(res) == 1
    assert res[0].node.get_content() == docs[0].text

    remove_docs(index, index.doc_ids)


def test_file_upload() -> None:
    try:
        index = VectaraIndex()
    except ValueError:
        pytest.skip("Missing Vectara credentials, skipping test")

    file_path = "docs/examples/data/paul_graham/paul_graham_essay.txt"
    id = index.insert_file(file_path)

    assert isinstance(index, VectaraIndex)

    # test query with Vectara summarization (default)
    query_engine = index.as_query_engine(similarity_top_k=3)
    res = query_engine.query("What software did Paul Graham write?")
    assert "paul graham" in str(res).lower() and "software" in str(res).lower()

    # test query with VectorStoreQuery (using OpenAI for summarization)
    query_engine = index.as_query_engine(similarity_top_k=3, summary_enabled=False)
    res = query_engine.query("What software did Paul Graham write?")
    assert "paul graham" in str(res).lower() and "software" in str(res).lower()

    remove_docs(index, [id])
