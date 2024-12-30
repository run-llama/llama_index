"""Optimization related classes and functions."""

import logging
from typing import Callable, List, Optional

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.indices.query.embedding_utils import get_top_k_embeddings
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from llama_index.core.settings import Settings

logger = logging.getLogger(__name__)


class SentenceEmbeddingOptimizer(BaseNodePostprocessor):
    """Optimization of a text chunk given the query by shortening the input text."""

    percentile_cutoff: Optional[float] = Field(
        description="Percentile cutoff for the top k sentences to use."
    )
    threshold_cutoff: Optional[float] = Field(
        description="Threshold cutoff for similarity for each sentence to use."
    )

    _embed_model: BaseEmbedding = PrivateAttr()
    _tokenizer_fn: Callable[[str], List[str]] = PrivateAttr()

    context_before: Optional[int] = Field(
        description="Number of sentences before retrieved sentence for further context"
    )

    context_after: Optional[int] = Field(
        description="Number of sentences after retrieved sentence for further context"
    )

    def __init__(
        self,
        embed_model: Optional[BaseEmbedding] = None,
        percentile_cutoff: Optional[float] = None,
        threshold_cutoff: Optional[float] = None,
        tokenizer_fn: Optional[Callable[[str], List[str]]] = None,
        context_before: Optional[int] = None,
        context_after: Optional[int] = None,
    ):
        """Optimizer class that is passed into BaseGPTIndexQuery.

        Should be set like this:

        .. code-block:: python
        from llama_index.core.optimization.optimizer import Optimizer
        optimizer = SentenceEmbeddingOptimizer(
                        percentile_cutoff=0.5
                        this means that the top 50% of sentences will be used.
                        Alternatively, you can set the cutoff using a threshold
                        on the similarity score. In this case only sentences with a
                        similarity score higher than the threshold will be used.
                        threshold_cutoff=0.7
                        these cutoffs can also be used together.
                    )

        query_engine = index.as_query_engine(
            optimizer=optimizer
        )
        response = query_engine.query("<query_str>")
        """
        super().__init__(
            percentile_cutoff=percentile_cutoff,
            threshold_cutoff=threshold_cutoff,
            context_after=context_after,
            context_before=context_before,
        )
        self._embed_model = embed_model or Settings.embed_model
        if self._embed_model is None:
            try:
                from llama_index.embeddings.openai import (
                    OpenAIEmbedding,
                )  # pants: no-infer-dep

                self._embed_model = OpenAIEmbedding()
            except ImportError:
                raise ImportError(
                    "`llama-index-embeddings-openai` package not found, "
                    "please run `pip install llama-index-embeddings-openai`"
                )

        if tokenizer_fn is None:
            import nltk

            tokenizer = nltk.tokenize.PunktSentenceTokenizer()
            tokenizer_fn = tokenizer.tokenize
        self._tokenizer_fn = tokenizer_fn

    @classmethod
    def class_name(cls) -> str:
        return "SentenceEmbeddingOptimizer"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Optimize a node text given the query by shortening the node text."""
        if query_bundle is None:
            return nodes

        for node_idx in range(len(nodes)):
            text = nodes[node_idx].node.get_content(metadata_mode=MetadataMode.LLM)

            split_text = self._tokenizer_fn(text)

            if query_bundle.embedding is None:
                query_bundle.embedding = (
                    self._embed_model.get_agg_embedding_from_queries(
                        query_bundle.embedding_strs
                    )
                )

            text_embeddings = self._embed_model._get_text_embeddings(split_text)

            num_top_k = None
            threshold = None
            if self.percentile_cutoff is not None:
                num_top_k = int(len(split_text) * self.percentile_cutoff)
            if self.threshold_cutoff is not None:
                threshold = self.threshold_cutoff

            top_similarities, top_idxs = get_top_k_embeddings(
                query_embedding=query_bundle.embedding,
                embeddings=text_embeddings,
                similarity_fn=self._embed_model.similarity,
                similarity_top_k=num_top_k,
                embedding_ids=list(range(len(text_embeddings))),
                similarity_cutoff=threshold,
            )

            if len(top_idxs) == 0:
                raise ValueError("Optimizer returned zero sentences.")

            rangeMin, rangeMax = 0, len(split_text)

            if self.context_before is None:
                self.context_before = 1
            if self.context_after is None:
                self.context_after = 1

            top_sentences = [
                " ".join(
                    split_text[
                        max(idx - self.context_before, rangeMin) : min(
                            idx + self.context_after + 1, rangeMax
                        )
                    ]
                )
                for idx in top_idxs
            ]

            logger.debug(f"> Top {len(top_idxs)} sentences with scores:\n")
            if logger.isEnabledFor(logging.DEBUG):
                for idx in range(len(top_idxs)):
                    logger.debug(
                        f"{idx}. {top_sentences[idx]} ({top_similarities[idx]})"
                    )

            nodes[node_idx].node.set_content(" ".join(top_sentences))

        return nodes
