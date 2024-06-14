import os

import pytest
from llama_index.core import Document
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore

from llama_index.postprocessor.mixedbreadai_rerank import MixedbreadAIRerank


def test_class():
    names_of_base_classes = [b.__name__ for b in MixedbreadAIRerank.__mro__]
    assert BaseNodePostprocessor.__name__ in names_of_base_classes


@pytest.mark.skipif(
    os.environ.get("MXBAI_API_KEY") is None, reason="Mixedbread AI API key required"
)
def test_accuracy():
    texts = ["Mockingbird", "Moby-Dick"]
    query = "Moby-Dick"
    result = MixedbreadAIRerank(
        api_key=os.environ["MXBAI_API_KEY"], model="mixedbread-ai/mxbai-rerank-large-v1"
    ).postprocess_nodes(
        [NodeWithScore(node=Document(text=text)) for text in texts],
        query_str=query,
    )
    assert result
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(node, NodeWithScore) for node in result)
    assert all(isinstance(node.node, Document) for node in result)
    assert all(isinstance(node.score, float) for node in result)
    assert result[0].node.text == "Moby-Dick"
