"""Optimization related classes and functions."""
import logging
from abc import abstractmethod
from typing import Callable, List, Optional

from llama_index.embeddings.base import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.indices.query.embedding_utils import get_top_k_embeddings
from llama_index.indices.query.schema import QueryBundle

logger = logging.getLogger(__name__)


class BaseTokenUsageOptimizer:
    """Base class for optimizers that should be overwritten."""

    @abstractmethod
    def optimize(self, query_bundle: QueryBundle, text: str) -> str:
        """Optimize the input text given the query."""
        raise NotImplementedError("Not implemented yet.")


class SentenceEmbeddingOptimizer(BaseTokenUsageOptimizer):
    """Optimization of a text chunk given the query by shortening the input text."""

    def __init__(
        self,
        embed_model: Optional[BaseEmbedding] = None,
        percentile_cutoff: Optional[float] = None,
        threshold_cutoff: Optional[float] = None,
        tokenizer_fn: Optional[Callable[[str], List[str]]] = None,
    ):
        """Optimizer class that is passed into BaseGPTIndexQuery.

        Should be set like this:

        .. code-block:: python
        from llama_index.optimization.optimizer import Optimizer
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
        self.embed_model = embed_model or OpenAIEmbedding()
        self._percentile_cutoff = percentile_cutoff
        self._threshold_cutoff = threshold_cutoff

        if tokenizer_fn is None:
            import nltk.data

            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt")
            tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
            tokenizer_fn = tokenizer.tokenize
        self._tokenizer_fn = tokenizer_fn

    def optimize(self, query_bundle: QueryBundle, text: str) -> str:
        """Optimize a text chunk given the query by shortening the input text."""
        split_text = self._tokenizer_fn(text)

        start_embed_token_ct = self.embed_model.total_tokens_used
        if query_bundle.embedding is None:
            query_bundle.embedding = self.embed_model.get_agg_embedding_from_queries(
                query_bundle.embedding_strs
            )
        text_embeddings = self.embed_model._get_text_embeddings(split_text)
        num_top_k = None
        threshold = None
        if self._percentile_cutoff is not None:
            num_top_k = int(len(split_text) * self._percentile_cutoff)
        if self._threshold_cutoff is not None:
            threshold = self._threshold_cutoff
        top_similarities, top_idxs = get_top_k_embeddings(
            query_embedding=query_bundle.embedding,
            embeddings=text_embeddings,
            similarity_fn=self.embed_model.similarity,
            similarity_top_k=num_top_k,
            embedding_ids=list(range(len(text_embeddings))),
            similarity_cutoff=threshold,
        )
        net_embed_tokens = self.embed_model.total_tokens_used - start_embed_token_ct
        logger.info(
            f"> [optimize] Total embedding token usage: " f"{net_embed_tokens} tokens"
        )
        if len(top_idxs) == 0:
            raise ValueError("Optimizer returned zero sentences.")
        top_sentences = [split_text[i] for i in top_idxs]

        logger.debug(f"> Top {len(top_idxs)} sentences with scores:\n")
        if logger.isEnabledFor(logging.DEBUG):
            for i in range(len(top_idxs)):
                logger.debug(f"{i}. {top_sentences[i]} ({top_similarities[i]})")
        return " ".join(top_sentences)
