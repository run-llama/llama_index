"""LLM reranker."""

from typing import Callable, List, Optional, Tuple, Union
import logging

from llama_index.core.bridge.pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    SerializeAsAny,
)
from llama_index.core.indices.utils import (
    default_format_node_batch_fn,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.default_prompts import STRUCTURED_CHOICE_SELECT_PROMPT
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.settings import Settings


logger = logging.getLogger(__name__)
dispatcher = get_dispatcher(__name__)


class DocumentWithRelevance(BaseModel):
    """
    Document rankings as selected by model.
    """

    document_number: int = Field(
        description="The number of the document within the provided list"
    )
    # Put min/max as a json schema extra so that pydantic doesn't enforce it but the model sees it.
    # Doesn't need to be strictly enforced but is useful for the model.
    relevance: int = Field(
        description="Relevance score from 1-10 of the document to the given query - based on the document content",
        json_schema_extra={"minimum": 1, "maximum": 10},
    )


class DocumentRelevanceList(BaseModel):
    """
    List of documents with relevance scores.
    """

    documents: List[DocumentWithRelevance] = Field(
        description="List of documents with relevance scores"
    )


def default_parse_structured_choice_select_answer(
    document_relevance_list: DocumentRelevanceList, num_choices: int
) -> Tuple[List[int], List[int]]:
    """
    Parse the answer from the choice select prompt.
    """
    documents = [
        doc
        for doc in document_relevance_list.documents
        if doc.document_number <= num_choices
    ]
    doc_numbers = [doc.document_number for doc in documents]
    doc_relevance_scores = [doc.relevance for doc in documents]
    return doc_numbers, doc_relevance_scores


class StructuredLLMRerank(BaseNodePostprocessor):
    """Structured LLM-based reranker."""

    top_n: int = Field(description="Top N nodes to return.")
    choice_select_prompt: SerializeAsAny[BasePromptTemplate] = Field(
        description="Choice select prompt."
    )
    choice_batch_size: int = Field(description="Batch size for choice select.")
    llm: LLM = Field(description="The LLM to rerank with.")

    _document_relevance_list_cls: type = PrivateAttr()
    _format_node_batch_fn: Callable = PrivateAttr()
    _parse_choice_select_answer_fn: Callable = PrivateAttr()
    _raise_on_prediction_failure: bool = PrivateAttr()

    def __init__(
        self,
        llm: Optional[LLM] = None,
        choice_select_prompt: Optional[BasePromptTemplate] = None,
        choice_batch_size: int = 10,
        format_node_batch_fn: Optional[Callable] = None,
        parse_choice_select_answer_fn: Optional[Callable] = None,
        document_relevance_list_cls: Optional[type] = None,
        raise_on_structured_prediction_failure: bool = True,
        top_n: int = 10,
    ) -> None:
        choice_select_prompt = choice_select_prompt or STRUCTURED_CHOICE_SELECT_PROMPT

        llm = llm or Settings.llm
        if not llm.metadata.is_function_calling_model:
            logger.warning(
                "StructuredLLMRerank constructed with a non-function-calling LLM. This may not work as expected."
            )

        super().__init__(
            llm=llm,
            choice_select_prompt=choice_select_prompt,
            choice_batch_size=choice_batch_size,
            top_n=top_n,
        )
        self._format_node_batch_fn = (
            format_node_batch_fn or default_format_node_batch_fn
        )
        self._parse_choice_select_answer_fn = (
            parse_choice_select_answer_fn
            or default_parse_structured_choice_select_answer
        )
        self._document_relevance_list_cls = (
            document_relevance_list_cls or DocumentRelevanceList
        )
        self._raise_on_structured_prediction_failure = (
            raise_on_structured_prediction_failure
        )

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"choice_select_prompt": self.choice_select_prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "choice_select_prompt" in prompts:
            self.choice_select_prompt = prompts["choice_select_prompt"]

    @classmethod
    def class_name(cls) -> str:
        return "StructuredLLMRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        dispatcher.event(
            ReRankStartEvent(
                query=query_bundle,
                nodes=nodes,
                top_n=self.top_n,
                model_name=self.llm.metadata.model_name,
            )
        )

        if query_bundle is None:
            raise ValueError("Query bundle must be provided.")
        if len(nodes) == 0:
            return []

        initial_results: List[NodeWithScore] = []
        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.llm.metadata.model_name,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            for idx in range(0, len(nodes), self.choice_batch_size):
                nodes_batch = [
                    node.node for node in nodes[idx : idx + self.choice_batch_size]
                ]

                query_str = query_bundle.query_str
                fmt_batch_str = self._format_node_batch_fn(nodes_batch)
                # call each batch independently
                result: Union[BaseModel, str] = self.llm.structured_predict(
                    output_cls=self._document_relevance_list_cls,
                    prompt=self.choice_select_prompt,
                    context_str=fmt_batch_str,
                    query_str=query_str,
                )
                # in case structured prediction fails, a str of the raised exception is returned
                if isinstance(result, str):
                    if self._raise_on_structured_prediction_failure:
                        raise ValueError(
                            f"Structured prediction failed for nodes {idx} - {idx + self.choice_batch_size}: {result}"
                        )
                    logger.warning(
                        f"Structured prediction failed for nodes {idx} - {idx + self.choice_batch_size}: {result}"
                    )
                    # add all nodes with score 0
                    initial_results.extend(
                        [NodeWithScore(node=node, score=0.0) for node in nodes_batch]
                    )
                    continue

                raw_choices, relevances = self._parse_choice_select_answer_fn(
                    result, len(nodes_batch)
                )
                choice_idxs = [int(choice) - 1 for choice in raw_choices]
                choice_nodes = [nodes_batch[idx] for idx in choice_idxs]
                relevances = relevances or [1.0 for _ in choice_nodes]
                initial_results.extend(
                    [
                        NodeWithScore(node=node, score=relevance)
                        for node, relevance in zip(choice_nodes, relevances)
                    ]
                )

            reranked_nodes = sorted(
                initial_results, key=lambda x: x.score or 0.0, reverse=True
            )[: self.top_n]
            event.on_end(payload={EventPayload.NODES: reranked_nodes})

        dispatcher.event(ReRankEndEvent(nodes=reranked_nodes))
        return reranked_nodes
