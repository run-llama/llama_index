import pytest
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank
from llama_index.core.schema import NodeWithScore, Document


@pytest.mark.integration()
def test_basic() -> None:
    text = "Testing leads to failure, and failure leads to understanding."
    result = NVIDIARerank().postprocess_nodes(
        [NodeWithScore(node=Document(text=text))],
        query_str=text,
    )
    assert result
    assert isinstance(result, list)
    assert len(result) == 1
    assert all(isinstance(node, NodeWithScore) for node in result)
    assert all(isinstance(node.node, Document) for node in result)
    assert all(isinstance(node.score, float) for node in result)
    assert all(node.node.text == text for node in result)
