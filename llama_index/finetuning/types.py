"""Finetuning Engine."""

from abc import ABC, abstractmethod
from llama_index.llms.base import LLM
from llama_index.embeddings.base import BaseEmbedding
from typing import Any


class BaseLLMFinetuningEngine(ABC):
    """Base LLM finetuning engine."""

    @abstractmethod
    def finetune(self) -> None:
        """Goes off and does stuff."""

    @abstractmethod
    def get_finetuned_model(self, **model_kwargs: Any) -> LLM:
        """Gets finetuned model."""


class BaseEmbeddingFinetuningEngine(ABC):
    """Base Embedding finetuning engine."""

    @abstractmethod
    def finetune(self) -> None:
        """Goes off and does stuff."""

    @abstractmethod
    def get_finetuned_model(self, **model_kwargs: Any) -> BaseEmbedding:
        """Gets finetuned model."""
