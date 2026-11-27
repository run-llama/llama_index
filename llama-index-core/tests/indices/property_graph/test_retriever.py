from typing import Any, List

import pytest

from llama_index.core import Settings
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks import CallbackManager
from llama_index.core.indices.property_graph.retriever import PGRetriever
from llama_index.core.schema import IndexNode, NodeWithScore, QueryBundle, TextNode


class _FixedRetriever(BaseRetriever):
    """Sub-retriever that always returns the nodes it was built with."""

    def __init__(self, nodes: List[NodeWithScore]) -> None:
        self._nodes = nodes
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self._nodes


def _index_node_result() -> List[NodeWithScore]:
    return [NodeWithScore(node=IndexNode(text="pointer", index_id="a"), score=0.5)]


def _text_node_result() -> List[NodeWithScore]:
    return [NodeWithScore(node=TextNode(text="plain"), score=0.5)]


def _make_retriever(nodes: List[NodeWithScore], **kwargs: Any) -> PGRetriever:
    sub_retrievers: List[Any] = [_FixedRetriever(nodes)]
    return PGRetriever(sub_retrievers, use_async=False, **kwargs)


def test_base_retriever_state_is_initialized() -> None:
    retriever = _make_retriever(_index_node_result())

    assert retriever.object_map == {}


def test_object_map_resolves_index_node() -> None:
    target = TextNode(text="resolved", id_="resolved-id")
    retriever = _make_retriever(_index_node_result(), object_map={"a": target})

    nodes = retriever.retrieve("query")

    assert len(nodes) == 1
    assert nodes[0].node.node_id == "resolved-id"
    assert nodes[0].node.get_content() == "resolved"


def test_callback_manager_is_kept() -> None:
    cm = CallbackManager([])
    retriever = _make_retriever(_text_node_result(), callback_manager=cm)

    assert retriever.callback_manager is cm

    retriever.retrieve("query")

    assert retriever.callback_manager is cm


def test_callback_manager_defaults_to_settings() -> None:
    retriever = _make_retriever(_text_node_result())

    retriever.retrieve("query")

    assert retriever.callback_manager is Settings.callback_manager


def test_unknown_kwargs_are_accepted() -> None:
    # PropertyGraphIndex.as_retriever forwards the same kwargs to the
    # sub-retrievers and to PGRetriever, so extra keys must not raise.
    retriever = _make_retriever(_text_node_result(), similarity_top_k=3)

    nodes = retriever.retrieve("query")

    assert len(nodes) == 1
    assert nodes[0].node.get_content() == "plain"


@pytest.mark.asyncio
async def test_aretrieve_resolves_index_node() -> None:
    target = TextNode(text="resolved", id_="resolved-id")
    retriever = _make_retriever(_index_node_result(), object_map={"a": target})

    nodes = await retriever.aretrieve("query")

    assert len(nodes) == 1
    assert nodes[0].node.node_id == "resolved-id"
    assert nodes[0].node.get_content() == "resolved"
