from typing import List, Any, Dict, Union, Generator
from hashlib import md5

import pytest

from llama_index.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores import VectaraVectorStore
from llama_index.vector_stores.types import VectorStoreQuery
from llama_index.schema import Document, BaseNode


def get_docs() -> List[Dict]:
    inputs = [
        {
            "text": "This ia test text for Vectara integraion with LlamaIndex",
            "metadata": {"test_num": "1"},
        },
        {
            "text": "And now for something completely differen",
            "metadata": {"test_num": "2"},
        },
        {
            "text": "when 900 year you will be, look as good you will no",
            "metadata": {"test_num": "3"},
        },
    ]
    docs: List[Document] = []
    ids = []
    for i in inputs:
        doc = Document(
            text=i["text"],
            metadata=i["metadata"],
        )
        docs.append(doc)
        ids.append(doc.id_)
    return docs, ids


def remove_docs(vs, ids):
    for id in ids:
        vs.delete(id)


def test_simple_query() -> None:
    vs = VectaraVectorStore()
    docs, ids = get_docs()
    vs.add_documents(docs)

    assert isinstance(vs, VectaraVectorStore)
    q = VectorStoreQuery(
        query_str="how will I look?", query_embedding=None, similarity_top_k=1
    )
    res = vs.query(q)
    assert res.nodes
    assert len(res.nodes) == 1

    remove_docs(vs, ids)


def test_with_filter_query() -> None:
    vs = VectaraVectorStore()
    docs, ids = get_docs()
    vs.add_documents(docs)

    assert isinstance(vs, VectaraVectorStore)
    q = VectorStoreQuery(
        query_str="how will I look?",
        query_embedding=None,
        similarity_top_k=1,
        filters={"test_num": 1},
    )
    res = vs.query(q)
    assert res.nodes
    assert len(res.nodes) == 1

    remove_docs(vs, ids)
