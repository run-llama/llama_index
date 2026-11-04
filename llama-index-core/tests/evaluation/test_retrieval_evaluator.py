import pytest
from typing import List, Optional

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.evaluation import RetrieverEvaluator, MultiModalRetrieverEvaluator
from llama_index.core.evaluation.retrieval.evaluator import RetrievalEvalMode
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode, ImageNode


class MockRetriever(BaseRetriever):
    """Mock retriever returning both text and image nodes for evaluation tests."""

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return [
            NodeWithScore(node=TextNode(id_="node_1", text="text 1"), score=0.9),
            NodeWithScore(node=ImageNode(id_="image_1", image="img.png"), score=0.8),
        ]

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return [
            NodeWithScore(node=TextNode(id_="node_1", text="text 1"), score=0.9),
            NodeWithScore(node=ImageNode(id_="image_1", image="img.png"), score=0.8),
        ]


class _TrackingPostprocessor(BaseNodePostprocessor):
    called_sync: bool = False
    called_async: bool = False

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        self.called_sync = True
        return nodes

    async def _apostprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        self.called_async = True
        return nodes


class _SyncOnlyPostprocessor(BaseNodePostprocessor):
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        for n in nodes:
            if type(n.node) is TextNode:
                n.node.text = f"processed_{n.node.text}"
        return nodes


@pytest.mark.asyncio
async def test_retriever_evaluator_async_postprocessor() -> None:
    postprocessor = _TrackingPostprocessor()
    evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr"],
        retriever=MockRetriever(),
        node_postprocessors=[postprocessor],
    )
    result = await evaluator.aevaluate(query="test", expected_ids=["node_1"])

    assert postprocessor.called_async is True
    assert postprocessor.called_sync is False
    assert result.retrieved_ids == ["node_1", "image_1"]


def test_retriever_evaluator_evaluate_calls_async_postprocessor() -> None:
    postprocessor = _TrackingPostprocessor()
    evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr"],
        retriever=MockRetriever(),
        node_postprocessors=[postprocessor],
    )
    result = evaluator.evaluate(query="test", expected_ids=["node_1"])

    # evaluate() wraps aevaluate() using asyncio_run, so apostprocess_nodes is called
    assert postprocessor.called_async is True
    assert postprocessor.called_sync is False
    assert result.retrieved_ids == ["node_1", "image_1"]


@pytest.mark.asyncio
async def test_retriever_evaluator_async_fallback() -> None:
    postprocessor = _SyncOnlyPostprocessor()
    evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr"],
        retriever=MockRetriever(),
        node_postprocessors=[postprocessor],
    )
    result = await evaluator.aevaluate(query="test", expected_ids=["node_1"])

    assert result.retrieved_texts == ["processed_text 1", ""]


@pytest.mark.asyncio
async def test_multimodal_retriever_evaluator_async_postprocessor() -> None:
    postprocessor = _TrackingPostprocessor()
    evaluator = MultiModalRetrieverEvaluator.from_metric_names(
        ["mrr"],
        retriever=MockRetriever(),
        node_postprocessors=[postprocessor],
    )
    result = await evaluator.aevaluate(
        query="test",
        expected_ids=["node_1"],
        mode=RetrievalEvalMode.TEXT,
    )

    assert postprocessor.called_async is True
    assert postprocessor.called_sync is False
    assert result.retrieved_ids == ["node_1", "image_1"]


@pytest.mark.asyncio
async def test_multimodal_retriever_evaluator_async_image_mode() -> None:
    postprocessor = _TrackingPostprocessor()
    evaluator = MultiModalRetrieverEvaluator.from_metric_names(
        ["mrr"],
        retriever=MockRetriever(),
        node_postprocessors=[postprocessor],
    )
    result = await evaluator.aevaluate(
        query="test",
        expected_ids=["image_1"],
        mode=RetrievalEvalMode.IMAGE,
    )

    assert postprocessor.called_async is True
    assert postprocessor.called_sync is False
    assert result.retrieved_ids == ["image_1"]
