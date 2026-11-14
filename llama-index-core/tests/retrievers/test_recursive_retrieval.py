import asyncio
from typing import Dict, List, Optional

import pytest

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import IndexNode, NodeWithScore, QueryBundle, TextNode


class StaticRetriever(BaseRetriever):
    def __init__(
        self,
        nodes: List[NodeWithScore],
        object_map: Optional[Dict[str, BaseRetriever]] = None,
    ) -> None:
        super().__init__(object_map=object_map)
        self.nodes = nodes

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self.nodes


def index_reference(index_id: str) -> NodeWithScore:
    return NodeWithScore(node=IndexNode(text=index_id, index_id=index_id))


@pytest.mark.asyncio
async def test_recursive_retrieval_is_concurrent_and_ordered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = StaticRetriever([NodeWithScore(node=TextNode(id_="first", text="first"))])
    second = StaticRetriever(
        [NodeWithScore(node=TextNode(id_="second", text="second"))]
    )
    first_started = asyncio.Event()
    second_finished = asyncio.Event()
    completion_order: List[str] = []

    async def retrieve_first(query_bundle: QueryBundle) -> List[NodeWithScore]:
        first_started.set()
        await second_finished.wait()
        completion_order.append("first")
        return first.nodes

    async def retrieve_second(query_bundle: QueryBundle) -> List[NodeWithScore]:
        await first_started.wait()
        completion_order.append("second")
        second_finished.set()
        return second.nodes

    monkeypatch.setattr(first, "_aretrieve", retrieve_first)
    monkeypatch.setattr(second, "_aretrieve", retrieve_second)
    retriever = StaticRetriever(
        [index_reference("first"), index_reference("second")],
        object_map={"first": first, "second": second},
    )

    # Events require overlap; the timeout only bounds a broken sequential run.
    nodes = await asyncio.wait_for(retriever.aretrieve("query"), timeout=5)

    assert completion_order == ["second", "first"]
    assert [node.node_id for node in nodes] == ["first", "second"]


@pytest.mark.asyncio
async def test_recursive_retrieval_preserves_passthrough_and_first_duplicate() -> None:
    first = StaticRetriever(
        [NodeWithScore(node=TextNode(id_="shared", text="first value"), score=0.25)]
    )
    duplicate = NodeWithScore(
        node=TextNode(id_="shared", text="later value"), score=0.9
    )
    unresolved = index_reference("unresolved")
    second = StaticRetriever(
        [NodeWithScore(node=TextNode(id_="unique", text="unique"), score=0.8)]
    )
    retriever = StaticRetriever(
        [index_reference("first"), duplicate, unresolved, index_reference("second")],
        object_map={"first": first, "second": second},
    )

    nodes = await retriever.aretrieve("query")

    assert [node.node_id for node in nodes] == ["shared", unresolved.node_id, "unique"]
    assert nodes[0].node.get_content() == "first value"
    assert nodes[0].score == 0.25
    assert nodes[1] is unresolved
    assert nodes[2].score == 0.8


@pytest.mark.asyncio
async def test_recursive_retrieval_propagates_nested_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = StaticRetriever([])

    async def fail(query_bundle: QueryBundle) -> List[NodeWithScore]:
        raise ValueError("nested retrieval failed")

    monkeypatch.setattr(child, "_aretrieve", fail)
    retriever = StaticRetriever([index_reference("child")], object_map={"child": child})

    with pytest.raises(ValueError, match="nested retrieval failed"):
        await retriever.aretrieve("query")
