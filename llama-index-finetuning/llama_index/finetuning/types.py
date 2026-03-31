"""Finetuning Engine."""

from abc import ABC, abstractmethod
from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.postprocessor.cohere_rerank import CohereRerank


class BaseLLMFinetuneEngine(ABC):
    """Base LLM finetuning engine."""

    @abstractmethod
    def finetune(self) -> None:
        """Run the finetuning process.

        This method should initiate the full finetuning loop, including
        loading training data, configuring the model, and saving the
        resulting checkpoint. Implementations may block until training
        is complete or launch an async job depending on the backend.
        """

    @abstractmethod
    def get_finetuned_model(self, **model_kwargs: Any) -> LLM:
        """Return the finetuned LLM.

        Args:
            **model_kwargs: Additional keyword arguments passed to the
                underlying model constructor.

        Returns:
            An LLM instance loaded from the finetuned checkpoint.
        """


class BaseEmbeddingFinetuneEngine(ABC):
    """Base Embedding finetuning engine."""

    @abstractmethod
    def finetune(self) -> None:
        """Run the finetuning process.

        This method should initiate the full finetuning loop, including
        loading training data, configuring the embedding model, and saving
        the resulting checkpoint.
        """

    @abstractmethod
    def get_finetuned_model(self, **model_kwargs: Any) -> BaseEmbedding:
        """Return the finetuned embedding model.

        Args:
            **model_kwargs: Additional keyword arguments passed to the
                underlying embedding model constructor.

        Returns:
            A BaseEmbedding instance loaded from the finetuned checkpoint.
        """


class BaseCrossEncoderFinetuningEngine(ABC):
    """Base Cross Encoder Finetuning Engine."""

    @abstractmethod
    def finetune(self) -> None:
        """Run the cross-encoder finetuning process.

        This method should initiate the full finetuning loop for a
        cross-encoder re-ranker, including loading pairwise training data
        and saving the resulting checkpoint.
        """

    @abstractmethod
    def get_finetuned_model(
        self, model_name: str, top_n: int = 3
    ) -> SentenceTransformerRerank:
        """Return the finetuned cross-encoder model as a re-ranker.

        Args:
            model_name: Name or path of the finetuned model.
            top_n: Number of top results to return during re-ranking.

        Returns:
            A SentenceTransformerRerank instance using the finetuned checkpoint.
        """


class BaseCohereRerankerFinetuningEngine(ABC):
    """Base Cohere Reranker Finetuning Engine."""

    @abstractmethod
    def finetune(self) -> None:
        """Run the Cohere reranker finetuning process.

        This method should initiate the full finetuning loop against the
        Cohere finetuning API, including uploading training data and
        waiting for the finetuned model to be ready.
        """

    @abstractmethod
    def get_finetuned_model(self, top_n: int = 5) -> CohereRerank:
        """Return the finetuned Cohere reranker model.

        Args:
            top_n: Number of top results to return during re-ranking.

        Returns:
            A CohereRerank instance using the finetuned model.
        """
