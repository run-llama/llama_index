from typing import List, Tuple

import pytest

from llama_index.indices.managed.vectara.base import VectaraIndex
from llama_index.schema import Document


def get_docs() -> Tuple[List[Document], List[str]]:
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
            "text": "when 900 year you will be, look as good you will not",
            "metadata": {"test_num": "3"},
        },
    ]
    docs: List[Document] = []
    ids = []
    for inp in inputs:
        doc = Document(
            text=inp["text"],
            metadata=inp["metadata"],
        )
        docs.append(doc)
        ids.append(doc.id_)
    return docs, ids


def remove_docs(index: VectaraIndex, ids: List) -> None:
    for id in ids:
        index.delete_ref_doc(id)


def test_simple_query() -> None:
    docs, ids = get_docs()
    try:
        index = VectaraIndex.from_documents(docs)
    except ValueError:
        pytest.skip("Missing Vectara credentials, skipping test")

    assert isinstance(index, VectaraIndex)
    qe = index.as_retriever(similarity_top_k=1)
    res = qe.retrieve("how will I look?")
    assert len(res) == 1
    assert res[0].node.text == docs[2].text

    remove_docs(index, ids)


def test_with_filter_query() -> None:
    docs, ids = get_docs()
    try:
        index = VectaraIndex.from_documents(docs)
    except ValueError:
        pytest.skip("Missing Vectara credentials, skipping test")

    assert isinstance(index, VectaraIndex)
    qe = index.as_retriever(similarity_top_k=1, filter="doc.test_num = '1'")
    res = qe.retrieve("how will I look?")
    assert len(res) == 1
    assert res[0].node.text == docs[0].text

    remove_docs(index, ids)


def test_file_upload() -> None:
    try:
        index = VectaraIndex()
    except ValueError:
        pytest.skip("Missing Vectara credentials, skipping test")

    file_path = "docs/examples/data/paul_graham/paul_graham_essay.txt"
    id = index.insert_file(file_path)

    assert isinstance(index, VectaraIndex)
    query_engine = index.as_query_engine(similarity_top_k=3)
    res = query_engine.query("What is a Manager Schedule?")
    assert "a manager schedule" in str(res).lower()

    remove_docs(index, [id])
