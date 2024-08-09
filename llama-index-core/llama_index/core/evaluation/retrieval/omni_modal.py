import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable, Collection, Dict, Generic, List, Optional, Tuple

from llama_index.core.async_utils import asyncio_run
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.embeddings.omni_modal_base import KD, KQ
from llama_index.core.evaluation.retrieval.metrics import resolve_metrics
from llama_index.core.evaluation.retrieval.metrics_base import (
    BaseRetrievalMetric,
    RetrievalMetricResult,
)
from llama_index.core.indices.omni_modal import OmniModalVectorIndexRetriever
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import BaseNode, QueryBundle


class OmniModalEmbeddingQAFinetuneDataset(BaseModel):
    """Omni-Modal Embedding QA Finetuning Dataset.

    Args:
        queries (Dict[str, QueryBundle]): Dict id -> query.
        corpus (Dict[str, BaseNode]): Dict id -> string.
        relevant_docs (Dict[str, List[str]]): Dict query id -> list of doc ids.

    """

    queries: Dict[str, QueryBundle]  # dict id -> query
    corpus: Dict[str, BaseNode]  # dict id -> string
    relevant_docs: Dict[str, List[str]]  # query id -> list of doc ids

    @property
    def query_docid_pairs(self) -> List[Tuple[QueryBundle, List[str]]]:
        """Get query, relevant doc ids."""
        return [
            (query, self.relevant_docs[query_id])
            for query_id, query in self.queries.items()
        ]

    def save_json(self, path: str) -> None:
        """Save json."""
        data = {
            "queries": {
                k: v.to_dict(encode_json=True) for k, v in self.queries.items()
            },
            "corpus": {k: v.to_dict() for k, v in self.corpus.items()},
            "relevant_docs": self.relevant_docs,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def from_json(
        cls,
        path: str,
        *,
        query_loader: Callable[[Dict[str, Any]], QueryBundle] = QueryBundle.from_dict,
        node_loader: Callable[[Dict[str, Any]], BaseNode] = BaseNode.from_dict,
    ) -> "OmniModalEmbeddingQAFinetuneDataset":
        """Load json."""
        with open(path) as f:
            data = json.load(f)

        return cls(
            queries={k: query_loader(v) for k, v in data["queries"].items()},
            corpus={k: node_loader(v) for k, v in data["corpus"].items()},
            relevant_docs=data["relevant_docs"],
        )


class OmniModalRetrievalEvalResult(BaseModel):
    """Retrieval eval result.

    NOTE: this abstraction might change in the future.

    Attributes:
        query_bundle (QueryBundle): Input query
        expected_ids (List[str]): Expected ids
        retrieved_ids (List[str]): Retrieved ids
        metric_dict (Dict[str, BaseRetrievalMetric]): \
            Metric dictionary for the evaluation

    """

    class Config:
        arbitrary_types_allowed = True

    query_bundle: QueryBundle = Field(..., description="Input query")
    query_type: str = Field(..., description="Modality type of query")
    expected_ids: List[str] = Field(..., description="Expected ids")
    expected_docs: Optional[List[BaseNode]] = Field(
        default=None,
        description="Expected documents associated with nodes provided in `expected_ids`",
    )
    retrieved_ids: List[str] = Field(..., description="Retrieved ids")
    retrieved_docs: List[BaseNode] = Field(..., description="Retrieved documents")
    doc_types: Optional[Collection[str]] = Field(
        default=None,
        description="Modality types of documents to match. `None` means all.",
    )
    metric_dict: Dict[str, RetrievalMetricResult] = Field(
        ..., description="Metric dictionary for the evaluation"
    )

    @property
    def metric_vals_dict(self) -> Dict[str, float]:
        """Dictionary of metric values."""
        return {k: v.score for k, v in self.metric_dict.items()}

    def __str__(self) -> str:
        """String representation."""
        return f"Query: {self.query_bundle}\n" f"Metrics: {self.metric_vals_dict!s}\n"


@dataclass
class OmniModalRetrievalEvaluator(Generic[KD, KQ]):
    """Omni-Modal Retrieval Evaluator class."""

    metrics: List[BaseRetrievalMetric]
    """List of metrics to evaluate"""

    retriever: OmniModalVectorIndexRetriever[KD, KQ]
    """Retriever to evaluate"""

    node_postprocessors: Optional[List[BaseNodePostprocessor]] = None
    """"Optional post-processor"""

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_metric_names(
        cls, metric_names: List[str], **kwargs: Any
    ) -> "OmniModalRetrievalEvaluator":
        """Create evaluator from metric names.

        Args:
            metric_names (List[str]): List of metric names
            **kwargs: Additional arguments for the evaluator

        """
        metric_types = resolve_metrics(metric_names)
        return cls(metrics=[metric() for metric in metric_types], **kwargs)

    async def _aget_retrieved_ids_and_docs(
        self,
        query_bundle: QueryBundle,
        *,
        query_type: KQ,
        # Defaults to all document modalities
        doc_types: Optional[Collection[KD]] = None,
    ) -> Tuple[List[str], List[BaseNode]]:
        """Get retrieved ids and documents."""
        scored_nodes = await self.retriever.aretrieve_multi_modal(
            query_bundle,
            query_type=query_type,
            doc_types=doc_types,
        )

        if self.node_postprocessors:
            for node_postprocessor in self.node_postprocessors:
                scored_nodes = node_postprocessor.postprocess_nodes(
                    scored_nodes,
                    query_bundle=query_bundle,
                )

        retrieved_nodes = [scored_node.node for scored_node in scored_nodes]

        return (
            [node.node_id for node in retrieved_nodes],
            retrieved_nodes,
        )

    def _compute_metrics(
        self,
        query_bundle: QueryBundle,
        *,
        retrieved_ids: List[str],
        retrieved_docs: List[BaseNode],
        expected_ids: List[str],
        expected_docs: Optional[List[BaseNode]],
    ) -> Dict[str, RetrievalMetricResult]:
        return {
            metric.metric_name: metric.compute(
                expected_ids=expected_ids,
                retrieved_ids=retrieved_ids,
                # Assume that the metric does not require this
                query=None,
                expected_texts=None,
                retrieved_texts=None,
            )
            for metric in self.metrics
        }

    def evaluate(
        self,
        query_bundle: QueryBundle,
        query_type: KQ,
        expected_ids: List[str],
        expected_docs: Optional[List[BaseNode]] = None,
        # Defaults to all document modalities
        doc_types: Optional[Collection[KD]] = None,
    ) -> OmniModalRetrievalEvalResult:
        """Run evaluation results with query string and expected ids.

        Args:
            query (str): Query string
            expected_ids (List[str]): Expected ids

        Returns:
            OmniModalRetrievalEvalResult: Evaluation result

        """
        return asyncio_run(
            self.aevaluate(
                query_bundle=query_bundle,
                query_type=query_type,
                expected_ids=expected_ids,
                expected_docs=expected_docs,
                doc_types=doc_types,
            )
        )

    async def aevaluate(
        self,
        query_bundle: QueryBundle,
        query_type: KQ,
        expected_ids: List[str],
        expected_docs: Optional[List[BaseNode]] = None,
        # Defaults to all document modalities
        doc_types: Optional[Collection[KD]] = None,
    ) -> OmniModalRetrievalEvalResult:
        """Run evaluation with query string, retrieved contexts,
        and generated response string.

        Subclasses can override this method to provide custom evaluation logic and
        take in additional arguments.
        """
        retrieved_ids, retrieved_docs = await self._aget_retrieved_ids_and_docs(
            query_bundle,
            query_type=query_type,
            doc_types=doc_types,
        )

        return OmniModalRetrievalEvalResult(
            query_bundle=query_bundle,
            query_type=query_type,
            expected_ids=expected_ids,
            expected_docs=expected_docs,
            retrieved_ids=retrieved_ids,
            retrieved_docs=retrieved_docs,
            doc_types=doc_types,
            metric_dict=self._compute_metrics(
                query_bundle,
                retrieved_ids=retrieved_ids,
                retrieved_docs=retrieved_docs,
                expected_ids=expected_ids,
                expected_docs=expected_docs,
            ),
        )

    async def aevaluate_dataset(
        self,
        dataset: OmniModalEmbeddingQAFinetuneDataset,
        query_type: KQ,
        *,
        # Defaults to all document modalities
        doc_types: Optional[Collection[KD]] = None,
        workers: int = 2,
        show_progress: bool = False,
    ) -> List[OmniModalRetrievalEvalResult]:
        """Run evaluation with dataset."""
        semaphore = asyncio.Semaphore(workers)

        async def eval_worker(
            query_bundle: QueryBundle,
            expected_ids: List[str],
            expected_docs: List[BaseNode],
        ) -> OmniModalRetrievalEvalResult:
            async with semaphore:
                ret_ids, ret_docs = await self._aget_retrieved_ids_and_docs(
                    query_bundle,
                    query_type=query_type,
                    doc_types=doc_types,
                )

                assert all(doc_id in dataset.corpus for doc_id in ret_ids), (
                    "Some retrieved documents do not belong in the dataset. "
                    "Make sure the dataset and retriever are built from the "
                    "same index."
                )

                return OmniModalRetrievalEvalResult(
                    query_bundle=query_bundle,
                    query_type=query_type,
                    expected_ids=expected_ids,
                    expected_docs=expected_docs,
                    retrieved_ids=ret_ids,
                    retrieved_docs=ret_docs,
                    doc_types=doc_types,
                    metric_dict=self._compute_metrics(
                        query_bundle,
                        retrieved_ids=ret_ids,
                        retrieved_docs=ret_docs,
                        expected_ids=expected_ids,
                        expected_docs=expected_docs,
                    ),
                )

        response_jobs = []
        for query_id, query in dataset.queries.items():
            expected_ids = dataset.relevant_docs[query_id]
            expected_docs = [dataset.corpus[doc_id] for doc_id in expected_ids]
            response_jobs.append(eval_worker(query, expected_ids, expected_docs))
        if show_progress:
            from tqdm.asyncio import tqdm_asyncio

            eval_results = await tqdm_asyncio.gather(
                *response_jobs,
                desc="Evaluating retrieval",
            )
        else:
            eval_results = await asyncio.gather(*response_jobs)

        return eval_results
