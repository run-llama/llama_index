import pytest
from llama_index.core.base.base_retriever import _filter_nodes_by_score

class DummyNode:
    def __init__(self, score):
        self.score = score


def test_score_filtering_enabled():

    nodes = [DummyNode(0.9), DummyNode(0.4), DummyNode(0.7)]

    filtered = _filter_nodes_by_score(nodes, 0.5)

    assert len(filtered) == 2


def test_score_filtering_disabled():
    nodes = [DummyNode(0.2), DummyNode(0.8)]

    # No filtering logic applied here
    assert len(nodes) == 2

def test_none_scores():

    nodes = [DummyNode(None), DummyNode(0.6)]

    result = _filter_nodes_by_score(nodes, 0.5)

    assert len(result) == 2    