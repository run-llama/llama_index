from typing import List, Any, Dict, Union, Generator
from hashlib import md5

import pytest

from llama_index.indices.managed.vectara.base import VectaraIndex
from llama_index.indices.managed.vectara.retriever import VectaraQuery
from llama_index.schema import Document


def get_docs() -> List[Dict]:
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


def remove_docs(vs, ids):
    for id in ids:
        vs.delete_ref_doc(id)


def add_documents(vs, docs):
    for doc in docs:
        vs.insert(doc)


def test_simple_query() -> None:
    vs = VectaraIndex()
    docs, ids = get_docs()
    add_documents(vs, docs)

    assert isinstance(vs, VectaraIndex)
    q = VectaraQuery(query_str="how will I look?", similarity_top_k=1)
    qe = vs.as_retriever()
    res = qe.retrieve(q)
    assert len(res) == 1
    assert res[0].node.text == docs[2].text

    remove_docs(vs, ids)


def test_with_filter_query() -> None:
    vs = VectaraIndex()
    docs, ids = get_docs()
    add_documents(vs, docs)

    assert isinstance(vs, VectaraIndex)
    q = VectaraQuery(
        query_str="how will I look?",
        similarity_top_k=1,
        filter="doc.test_num = '1'",
    )
    qe = vs.as_retriever()
    res = qe.retrieve(q)
    assert len(res) == 1
    assert res[0].node.text == docs[0].text

    remove_docs(vs, ids)
