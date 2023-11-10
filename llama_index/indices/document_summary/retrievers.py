"""Document summary retrievers.

This module contains retrievers for document summary indices.

"""

import logging
from typing import Any, Callable, List, Optional

from llama_index.core import BaseRetriever
from llama_index.indices.document_summary.base import DocumentSummaryIndex
from llama_index.indices.utils import (
    default_format_node_batch_fn,
    default_parse_choice_select_answer_fn,
)
from llama_index.prompts import BasePromptTemplate
from llama_index.prompts.default_prompts import DEFAULT_CHOICE_SELECT_PROMPT
from llama_index.schema import NodeWithScore, QueryBundle
from llama_index.service_context import ServiceContext
from llama_index.vector_stores.types import VectorStoreQuery

logger = logging.getLogger(__name__)


class DocumentSummaryIndexLLMRetriever(BaseRetriever):
    """Document Summary Index LLM Retriever.

    By default, select relevant summaries from index using LLM calls.

    Args:
        index (DocumentSummaryIndex): The index to retrieve from.
        choice_select_prompt (Optional[BasePromptTemplate]): The prompt to use for selecting relevant summaries.
        choice_batch_size (int): The number of summary nodes to send to LLM at a time.
        choice_top_k (int): The number of summary nodes to retrieve.
        format_node_batch_fn (Callable): Function to format a batch of nodes for LLM.
        parse_choice_select_answer_fn (Callable): Function to parse LLM response.
        service_context (ServiceContext): The service context to use.
    """

    def __init__(
        self,
        index: DocumentSummaryIndex,
        choice_select_prompt: Optional[BasePromptTemplate] = None,
        choice_batch_size: int = 10,
        choice_top_k: int = 1,
        format_node_batch_fn: Optional[Callable] = None,
        parse_choice_select_answer_fn: Optional[Callable] = None,
        service_context: Optional[ServiceContext] = None,
        **kwargs: Any,
    ) -> None:
        self._index = index
        self._choice_select_prompt = (
            choice_select_prompt or DEFAULT_CHOICE_SELECT_PROMPT
        )
        self._choice_batch_size = choice_batch_size
        self._choice_top_k = choice_top_k
        self._format_node_batch_fn = (
            format_node_batch_fn or default_format_node_batch_fn
        )
        self._parse_choice_select_answer_fn = (
            parse_choice_select_answer_fn or default_parse_choice_select_answer_fn
        )
        self._service_context = service_context or index.service_context

    def _retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        """Retrieve nodes."""
        summary_ids = self._index.index_struct.summary_ids

        all_summary_ids: List[str] = []
        all_relevances: List[float] = []
        for idx in range(0, len(summary_ids), self._choice_batch_size):
            summary_ids_batch = summary_ids[idx : idx + self._choice_batch_size]
            summary_nodes = self._index.docstore.get_nodes(summary_ids_batch)
            query_str = query_bundle.query_str
            fmt_batch_str = self._format_node_batch_fn(summary_nodes)
            # call each batch independently
            raw_response = self._service_context.llm_predictor.predict(
                self._choice_select_prompt,
                context_str=fmt_batch_str,
                query_str=query_str,
            )
            raw_choices, relevances = self._parse_choice_select_answer_fn(
                raw_response, len(summary_nodes)
            )
            choice_idxs = [choice - 1 for choice in raw_choices]

            choice_summary_ids = [summary_ids_batch[ci] for ci in choice_idxs]

            all_summary_ids.extend(choice_summary_ids)
            all_relevances.extend(relevances)

        zipped_list = list(zip(all_summary_ids, all_relevances))
        sorted_list = sorted(zipped_list, key=lambda x: x[1], reverse=True)
        top_k_list = sorted_list[: self._choice_top_k]

        results = []
        for summary_id, relevance in top_k_list:
            node_ids = self._index.index_struct.summary_id_to_node_ids[summary_id]
            nodes = self._index.docstore.get_nodes(node_ids)
            results.extend([NodeWithScore(node=n, score=relevance) for n in nodes])

        return results


class DocumentSummaryIndexEmbeddingRetriever(BaseRetriever):
    """Document Summary Index Embedding Retriever.

    Args:
        index (DocumentSummaryIndex): The index to retrieve from.
        similarity_top_k (int): The number of summary nodes to retrieve.

    """

    def __init__(
        self, index: DocumentSummaryIndex, similarity_top_k: int = 1, **kwargs: Any
    ) -> None:
        """Init params."""
        self._index = index
        self._vector_store = self._index.vector_store
        self._service_context = self._index.service_context
        self._docstore = self._index.docstore
        self._index_struct = self._index.index_struct

        self._similarity_top_k = similarity_top_k

    def _retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        """Retrieve nodes."""
        if self._vector_store.is_embedding_query:
            if query_bundle.embedding is None:
                query_bundle.embedding = (
                    self._service_context.embed_model.get_agg_embedding_from_queries(
                        query_bundle.embedding_strs
                    )
                )

        query = VectorStoreQuery(
            query_embedding=query_bundle.embedding,
            similarity_top_k=self._similarity_top_k,
        )
        query_result = self._vector_store.query(query)

        top_k_summary_ids: List[str]
        if query_result.ids is not None:
            top_k_summary_ids = query_result.ids
        elif query_result.nodes is not None:
            top_k_summary_ids = [n.node_id for n in query_result.nodes]
        else:
            raise ValueError(
                "Vector store query result should return "
                "at least one of nodes or ids."
            )

        results = []
        for summary_id in top_k_summary_ids:
            node_ids = self._index_struct.summary_id_to_node_ids[summary_id]
            nodes = self._docstore.get_nodes(node_ids)
            results.extend([NodeWithScore(node=n) for n in nodes])
        return results


# legacy, backward compatibility
DocumentSummaryIndexRetriever = DocumentSummaryIndexLLMRetriever
