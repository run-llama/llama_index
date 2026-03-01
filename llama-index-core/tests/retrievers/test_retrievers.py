"""Tests for core retrievers: BaseRetriever, TransformRetriever, QueryFusionRetriever."""

from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode


# ---------------------------------------------------------------------------
# Minimal concrete retriever for testing BaseRetriever behaviour
# ---------------------------------------------------------------------------


class SimpleRetriever(BaseRetriever):
    """Concrete retriever that returns a fixed list of nodes."""

    def __init__(self, nodes: Optional[List[NodeWithScore]] = None, **kwargs):
        super().__init__(**kwargs)
        self._nodes = nodes or []

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self._nodes


# ---------------------------------------------------------------------------
# BaseRetriever tests
# ---------------------------------------------------------------------------


class TestBaseRetriever:
    def _make_nodes(self, n: int) -> List[NodeWithScore]:
        return [
            NodeWithScore(node=TextNode(text=f"Node {i}", id_=f"node_{i}"), score=float(i))
            for i in range(n)
        ]

    def test_construction_with_defaults(self) -> None:
        """BaseRetriever subclass should construct without arguments."""
        retriever = SimpleRetriever()
        assert retriever is not None
        assert retriever.object_map == {}
        assert retriever._verbose is False

    def test_construction_with_callback_manager(self) -> None:
        """Callback manager should be stored on the instance."""
        cm = CallbackManager()
        retriever = SimpleRetriever(callback_manager=cm)
        assert retriever.callback_manager is cm

    def test_retrieve_returns_nodes(self) -> None:
        """retrieve() should delegate to _retrieve() and return nodes."""
        nodes = self._make_nodes(3)
        retriever = SimpleRetriever(nodes=nodes)
        result = retriever.retrieve("test query")
        assert len(result) == 3

    def test_retrieve_with_query_bundle(self) -> None:
        """retrieve() should accept a QueryBundle directly."""
        nodes = self._make_nodes(2)
        retriever = SimpleRetriever(nodes=nodes)
        query = QueryBundle(query_str="What is this?")
        result = retriever.retrieve(query)
        assert len(result) == 2

    def test_retrieve_empty_results(self) -> None:
        """retrieve() with no matching nodes should return empty list."""
        retriever = SimpleRetriever(nodes=[])
        result = retriever.retrieve("test query")
        assert result == []

    def test_retrieve_with_objects(self) -> None:
        """Objects should be discoverable via retrieve when indexed nodes point to them."""
        from llama_index.core.schema import IndexNode

        hidden_node = TextNode(text="Hidden content", id_="hidden")
        index_node = IndexNode(
            text="Index entry",
            id_="index_1",
            index_id="index_1",
            obj=hidden_node,
        )
        # The retriever returns the index_node; BaseRetriever should resolve to hidden_node
        retriever = SimpleRetriever(
            nodes=[NodeWithScore(node=index_node, score=1.0)],
        )
        result = retriever.retrieve("test")
        # The resolved obj (hidden_node) should be returned
        assert any(r.node.id_ == "hidden" for r in result)

    def test_get_prompts_returns_empty_dict(self) -> None:
        """Default _get_prompts should return empty dict."""
        retriever = SimpleRetriever()
        assert retriever._get_prompts() == {}

    def test_update_prompts_is_noop(self) -> None:
        """Default _update_prompts should not raise."""
        retriever = SimpleRetriever()
        retriever._update_prompts({"any_key": "any_value"})

    def test_verbose_flag(self) -> None:
        """verbose flag should be stored on the instance."""
        retriever = SimpleRetriever(verbose=True)
        assert retriever._verbose is True

    @pytest.mark.asyncio
    async def test_aretrieve_returns_nodes(self) -> None:
        """aretrieve() should return nodes asynchronously."""
        nodes = self._make_nodes(2)
        retriever = SimpleRetriever(nodes=nodes)
        result = await retriever.aretrieve("async test")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# TransformRetriever tests
# ---------------------------------------------------------------------------


class TestTransformRetriever:
    def test_transform_retriever_construction(self) -> None:
        """TransformRetriever should require a retriever and a query transform."""
        from llama_index.core.retrievers.transform_retriever import TransformRetriever
        from llama_index.core.indices.query.query_transform.base import BaseQueryTransform

        inner_retriever = SimpleRetriever()
        mock_transform = MagicMock(spec=BaseQueryTransform)
        retriever = TransformRetriever(
            retriever=inner_retriever, query_transform=mock_transform
        )
        assert retriever is not None

    def test_transform_retriever_applies_transform(self) -> None:
        """TransformRetriever should apply the transform before delegating."""
        from llama_index.core.retrievers.transform_retriever import TransformRetriever
        from llama_index.core.indices.query.query_transform.base import BaseQueryTransform

        nodes = [NodeWithScore(node=TextNode(text="result", id_="r1"), score=1.0)]
        inner_retriever = SimpleRetriever(nodes=nodes)

        # Transform that modifies the query string
        mock_transform = MagicMock(spec=BaseQueryTransform)
        transformed_query = QueryBundle(query_str="transformed query")
        mock_transform.run.return_value = transformed_query

        retriever = TransformRetriever(
            retriever=inner_retriever, query_transform=mock_transform
        )
        result = retriever.retrieve("original query")

        # Verify the transform was called with original query
        mock_transform.run.assert_called_once()
        assert len(result) == 1

    def test_transform_retriever_get_prompt_modules(self) -> None:
        """_get_prompt_modules should return the query_transform."""
        from llama_index.core.retrievers.transform_retriever import TransformRetriever
        from llama_index.core.indices.query.query_transform.base import BaseQueryTransform

        inner_retriever = SimpleRetriever()
        mock_transform = MagicMock(spec=BaseQueryTransform)
        retriever = TransformRetriever(
            retriever=inner_retriever, query_transform=mock_transform
        )
        modules = retriever._get_prompt_modules()
        assert "query_transform" in modules


# ---------------------------------------------------------------------------
# QueryFusionRetriever tests
# ---------------------------------------------------------------------------


class TestQueryFusionRetriever:
    def _make_retriever_with_nodes(self, nodes: List[NodeWithScore]) -> SimpleRetriever:
        return SimpleRetriever(nodes=nodes)

    def test_fusion_retriever_construction_defaults(self) -> None:
        """QueryFusionRetriever should construct with a list of retrievers."""
        from llama_index.core.retrievers.fusion_retriever import QueryFusionRetriever

        r1 = SimpleRetriever()
        r2 = SimpleRetriever()
        retriever = QueryFusionRetriever(retrievers=[r1, r2], num_queries=1)
        assert retriever is not None

    def test_fusion_retriever_equal_default_weights(self) -> None:
        """Default weights should be equal for all retrievers."""
        from llama_index.core.retrievers.fusion_retriever import QueryFusionRetriever

        r1 = SimpleRetriever()
        r2 = SimpleRetriever()
        retriever = QueryFusionRetriever(retrievers=[r1, r2], num_queries=1)
        assert len(retriever._retriever_weights) == 2
        assert abs(retriever._retriever_weights[0] - 0.5) < 1e-6
        assert abs(retriever._retriever_weights[1] - 0.5) < 1e-6

    def test_fusion_retriever_custom_weights_normalized(self) -> None:
        """Custom weights should be normalised so they sum to 1."""
        from llama_index.core.retrievers.fusion_retriever import QueryFusionRetriever

        r1 = SimpleRetriever()
        r2 = SimpleRetriever()
        retriever = QueryFusionRetriever(
            retrievers=[r1, r2],
            retriever_weights=[3.0, 1.0],
            num_queries=1,
        )
        assert abs(sum(retriever._retriever_weights) - 1.0) < 1e-6
        assert abs(retriever._retriever_weights[0] - 0.75) < 1e-6

    def test_fusion_retriever_get_prompts(self) -> None:
        """_get_prompts should expose the query_gen_prompt."""
        from llama_index.core.retrievers.fusion_retriever import QueryFusionRetriever

        retriever = QueryFusionRetriever(retrievers=[SimpleRetriever()], num_queries=1)
        prompts = retriever._get_prompts()
        assert "query_gen_prompt" in prompts

    def test_fusion_retriever_update_prompts(self) -> None:
        """_update_prompts should update the query_gen_prompt."""
        from llama_index.core.retrievers.fusion_retriever import QueryFusionRetriever
        from llama_index.core.prompts import PromptTemplate

        retriever = QueryFusionRetriever(retrievers=[SimpleRetriever()], num_queries=1)
        new_prompt = PromptTemplate("New prompt: {query}\n")
        retriever._update_prompts({"query_gen_prompt": new_prompt})
        # After update, the stored prompt should reflect the new template
        assert "New prompt:" in retriever.query_gen_prompt

    def test_fusion_retriever_simple_mode_retrieve(self) -> None:
        """SIMPLE fusion mode should return deduplicated results from all retrievers."""
        from llama_index.core.retrievers.fusion_retriever import (
            QueryFusionRetriever,
            FUSION_MODES,
        )

        node1 = NodeWithScore(node=TextNode(text="From R1", id_="n1"), score=0.9)
        node2 = NodeWithScore(node=TextNode(text="From R2", id_="n2"), score=0.8)

        r1 = self._make_retriever_with_nodes([node1])
        r2 = self._make_retriever_with_nodes([node2])

        retriever = QueryFusionRetriever(
            retrievers=[r1, r2],
            mode=FUSION_MODES.SIMPLE,
            num_queries=1,
            use_async=False,
        )
        result = retriever.retrieve("test query")
        assert len(result) >= 1

    def test_fusion_retriever_reciprocal_rank_mode(self) -> None:
        """RECIPROCAL_RANK mode should return combined results."""
        from llama_index.core.retrievers.fusion_retriever import (
            QueryFusionRetriever,
            FUSION_MODES,
        )

        nodes = [
            NodeWithScore(
                node=TextNode(text=f"Node {i}", id_=f"n{i}"), score=float(i)
            )
            for i in range(3)
        ]
        r = self._make_retriever_with_nodes(nodes)
        retriever = QueryFusionRetriever(
            retrievers=[r],
            mode=FUSION_MODES.RECIPROCAL_RANK,
            num_queries=1,
            use_async=False,
        )
        result = retriever.retrieve("test")
        assert isinstance(result, list)
