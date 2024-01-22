"""Retrievers for SummaryIndex."""
import logging
from typing import Any, Callable, List, Optional, Tuple

from llama_index.callbacks.base import CallbackManager
from llama_index.core.base_retriever import BaseRetriever
from llama_index.indices.list.base import SummaryIndex
from llama_index.indices.query.embedding_utils import get_top_k_embeddings
from llama_index.indices.utils import (
    default_format_node_batch_fn,
    default_parse_choice_select_answer_fn,
)
from llama_index.prompts import PromptTemplate
from llama_index.prompts.default_prompts import (
    DEFAULT_CHOICE_SELECT_PROMPT,
)
from llama_index.schema import BaseNode, MetadataMode, NodeWithScore, QueryBundle
from llama_index.service_context import ServiceContext

logger = logging.getLogger(__name__)


class SummaryIndexRetriever(BaseRetriever):
    """Simple retriever for SummaryIndex that returns all nodes.

    Args:
        index (SummaryIndex): The index to retrieve from.

    """

    def __init__(
        self,
        index: SummaryIndex,
        callback_manager: Optional[CallbackManager] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        self._index = index
        super().__init__(
            callback_manager=callback_manager, object_map=object_map, verbose=verbose
        )

    def _retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        """Retrieve nodes."""
        del query_bundle

        node_ids = self._index.index_struct.nodes
        nodes = self._index.docstore.get_nodes(node_ids)
        return [NodeWithScore(node=node) for node in nodes]


class SummaryIndexEmbeddingRetriever(BaseRetriever):
    """Embedding based retriever for SummaryIndex.

    Generates embeddings in a lazy fashion for all
    nodes that are traversed.

    Args:
        index (SummaryIndex): The index to retrieve from.
        similarity_top_k (Optional[int]): The number of top nodes to return.

    """

    def __init__(
        self,
        index: SummaryIndex,
        similarity_top_k: Optional[int] = 1,
        callback_manager: Optional[CallbackManager] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        self._index = index
        self._similarity_top_k = similarity_top_k
        super().__init__(
            callback_manager=callback_manager, object_map=object_map, verbose=verbose
        )

    def _retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        """Retrieve nodes."""
        node_ids = self._index.index_struct.nodes
        # top k nodes
        nodes = self._index.docstore.get_nodes(node_ids)
        query_embedding, node_embeddings = self._get_embeddings(query_bundle, nodes)

        top_similarities, top_idxs = get_top_k_embeddings(
            query_embedding,
            node_embeddings,
            similarity_top_k=self._similarity_top_k,
            embedding_ids=list(range(len(nodes))),
        )

        top_k_nodes = [nodes[i] for i in top_idxs]

        node_with_scores = []
        for node, similarity in zip(top_k_nodes, top_similarities):
            node_with_scores.append(NodeWithScore(node=node, score=similarity))

        logger.debug(f"> Top {len(top_idxs)} nodes:\n")
        nl = "\n"
        logger.debug(f"{ nl.join([n.get_content() for n in top_k_nodes]) }")
        return node_with_scores

    def _get_embeddings(
        self, query_bundle: QueryBundle, nodes: List[BaseNode]
    ) -> Tuple[List[float], List[List[float]]]:
        """Get top nodes by similarity to the query."""
        if query_bundle.embedding is None:
            query_bundle.embedding = (
                self._index._service_context.embed_model.get_agg_embedding_from_queries(
                    query_bundle.embedding_strs
                )
            )

        node_embeddings: List[List[float]] = []
        nodes_embedded = 0
        for node in nodes:
            if node.embedding is None:
                nodes_embedded += 1
                node.embedding = (
                    self._index.service_context.embed_model.get_text_embedding(
                        node.get_content(metadata_mode=MetadataMode.EMBED)
                    )
                )

            node_embeddings.append(node.embedding)
        return query_bundle.embedding, node_embeddings


class SummaryIndexLLMRetriever(BaseRetriever):
    """LLM retriever for SummaryIndex.

    Args:
        index (SummaryIndex): The index to retrieve from.
        choice_select_prompt (Optional[PromptTemplate]): A Choice-Select Prompt
           (see :ref:`Prompt-Templates`).)
        choice_batch_size (int): The number of nodes to query at a time.
        format_node_batch_fn (Optional[Callable]): A function that formats a
            batch of nodes.
        parse_choice_select_answer_fn (Optional[Callable]): A function that parses the
            choice select answer.
        service_context (Optional[ServiceContext]): A service context.

    """

    def __init__(
        self,
        index: SummaryIndex,
        choice_select_prompt: Optional[PromptTemplate] = None,
        choice_batch_size: int = 10,
        format_node_batch_fn: Optional[Callable] = None,
        parse_choice_select_answer_fn: Optional[Callable] = None,
        service_context: Optional[ServiceContext] = None,
        callback_manager: Optional[CallbackManager] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        self._index = index
        self._choice_select_prompt = (
            choice_select_prompt or DEFAULT_CHOICE_SELECT_PROMPT
        )
        self._choice_batch_size = choice_batch_size
        self._format_node_batch_fn = (
            format_node_batch_fn or default_format_node_batch_fn
        )
        self._parse_choice_select_answer_fn = (
            parse_choice_select_answer_fn or default_parse_choice_select_answer_fn
        )
        self._service_context = service_context or index.service_context
        super().__init__(
            callback_manager=callback_manager, object_map=object_map, verbose=verbose
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes."""
        node_ids = self._index.index_struct.nodes
        results = []
        for idx in range(0, len(node_ids), self._choice_batch_size):
            node_ids_batch = node_ids[idx : idx + self._choice_batch_size]
            nodes_batch = self._index.docstore.get_nodes(node_ids_batch)

            query_str = query_bundle.query_str
            fmt_batch_str = self._format_node_batch_fn(nodes_batch)
            # call each batch independently
            raw_response = self._service_context.llm.predict(
                self._choice_select_prompt,
                context_str=fmt_batch_str,
                query_str=query_str,
            )

            raw_choices, relevances = self._parse_choice_select_answer_fn(
                raw_response, len(nodes_batch)
            )
            choice_idxs = [int(choice) - 1 for choice in raw_choices]
            choice_node_ids = [node_ids_batch[idx] for idx in choice_idxs]

            choice_nodes = self._index.docstore.get_nodes(choice_node_ids)
            relevances = relevances or [1.0 for _ in choice_nodes]
            results.extend(
                [
                    NodeWithScore(node=node, score=relevance)
                    for node, relevance in zip(choice_nodes, relevances)
                ]
            )
        return results


# for backwards compatibility
ListIndexEmbeddingRetriever = SummaryIndexEmbeddingRetriever
ListIndexLLMRetriever = SummaryIndexLLMRetriever
ListIndexRetriever = SummaryIndexRetriever
