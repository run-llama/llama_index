import json
from requests import Response
from unittest import mock
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, Document
from llama_index.postprocessor.siliconflow_rerank import SiliconFlowRerank

_FAKE_API_KEY = ""
_FAKE_RERANK_RESPONSE = Response()
_FAKE_RERANK_RESPONSE._content = json.dumps(
    {
        "id": "<string>",
        "results": [
            {
                "document": {"text": "last 1"},
                "index": 2,
                "relevance_score": 0.9,
            },
            {
                "document": {"text": "last 2"},
                "index": 3,
                "relevance_score": 0.8,
            },
        ],
        "tokens": {"input_tokens": 123, "output_tokens": 123},
    }
).encode("utf-8")


def test_class():
    names_of_base_classes = [b.__name__ for b in SiliconFlowRerank.__mro__]
    assert BaseNodePostprocessor.__name__ in names_of_base_classes


def test_fake_rerank():
    input_nodes = [
        NodeWithScore(node=Document(doc_id="1", text="first 1")),
        NodeWithScore(node=Document(doc_id="2", text="first 2")),
        NodeWithScore(node=Document(doc_id="3", text="last 1")),
        NodeWithScore(node=Document(doc_id="4", text="last 2")),
    ]
    expected_nodes = [
        NodeWithScore(node=Document(doc_id="3", text="last 1"), score=0.9),
        NodeWithScore(node=Document(doc_id="4", text="last 2"), score=0.8),
    ]
    reranker = SiliconFlowRerank(api_key=_FAKE_API_KEY)

    with mock.patch.object(
        reranker._session,
        "post",
        return_value=_FAKE_RERANK_RESPONSE,
    ):
        actual_nodes = reranker.postprocess_nodes(input_nodes, query_str="last")
        assert actual_nodes == expected_nodes
