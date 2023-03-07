"""Optimization related classes and functions."""
import logging
from typing import Dict, Optional

from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.query.embedding_utils import get_top_k_embeddings
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.response.builder import TextChunk


class Optimizer:
    """Optimization of a text chunk given the query by shortening the input text."""

    def __init__(
        self,
        split_mode: str,
        comparison_mode: str,
        embed_model: Optional[BaseEmbedding] = None,
        cutoffs: Dict[str, float] = {},
    ):
        """Optimizer class that is passed into BaseGPTIndexQuery.

        Should be set like this:

        .. code-block:: python
        from gpt_index.optimization.optimizer import Optimizer
        optimizer = Optimizer(
                        split_mode="sentence",
                        comparison_mode="embedding",
                        cutoffs = {"percentile": 0.5}
                        # this means that the top 50% of sentences will be used.
                        # Alternatively, you can set the cutoff using a threshold
                        # on the similarity score.
                        # cutoffs = {"threshold": 0.7}
                    )

        response = index.query(
            "<query_str>", optimizer=optimizer
        )
        """
        self._split_mode = split_mode
        self._comparison_mode = comparison_mode
        self._embed_model = embed_model or OpenAIEmbedding()
        self._cutoffs = cutoffs

    def optimize(self, query_bundle: QueryBundle, textChunk: TextChunk) -> TextChunk:
        """Optimize a text chunk given the query by shortening the input text."""
        import nltk.data

        if self.is_valid_mode():

            if self._split_mode == "sentence":
                try:
                    nltk.data.find("tokenizers/punkt")
                except LookupError:
                    nltk.download("punkt")
                tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
                split_text = tokenizer.tokenize(textChunk.text)
                if self._comparison_mode == "embedding":
                    query_embedding = self._embed_model.get_agg_embedding_from_queries(
                        query_bundle.embedding_strs
                    )
                    text_embeddings = self._embed_model._get_text_embeddings(split_text)
                    num_top_k = None
                    threshold = None
                    if "percentile" in self._cutoffs:
                        percentile = self._cutoffs["percentile"]
                        num_top_k = int(len(split_text) * percentile)
                    if "threshold" in self._cutoffs:
                        threshold = self._cutoffs["threshold"]
                    top_similarities, top_idxs = get_top_k_embeddings(
                        query_embedding=query_embedding,
                        embeddings=text_embeddings,
                        similarity_fn=self._embed_model.similarity,
                        similarity_top_k=num_top_k,
                        embedding_ids=[i for i in range(len(text_embeddings))],
                        similarity_cutoff=threshold,
                    )
                    if len(top_idxs) == 0:
                        raise ValueError("Optimizer returned zero sentences.")
                    top_sentences = [split_text[i] for i in top_idxs]

                    logging.debug(f"> Top {len(top_idxs)} sentences with scores:\n")
                    for i in range(len(top_idxs)):
                        logging.debug(
                            f"{i}. {top_sentences[i]} ({top_similarities[i]})"
                        )
                    return TextChunk(text=" ".join(top_sentences))

            else:
                print("Invalid split mode")

        else:
            print("Invalid mode")
        return textChunk

    def is_valid_mode(self) -> bool:
        """Check if the mode is valid."""
        return self._split_mode in ["sentence"] and self._comparison_mode in [
            "embedding"
        ]
