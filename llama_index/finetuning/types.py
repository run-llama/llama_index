"""Finetuning Engine."""

from abc import ABC, abstractmethod
from llama_index.llms.base import LLM
from typing import Any


class BaseFinetuningEngine(ABC):
    """Base finetuning engine."""

    @abstractmethod
    def finetune(self) -> None:
        """Goes off and does stuff."""

    @abstractmethod
    def get_finetuned_model(self, **model_kwargs: Any) -> LLM:
        """Gets finetuned model."""
