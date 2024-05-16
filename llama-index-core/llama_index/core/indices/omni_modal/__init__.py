"""Decouples different modalities from each other, making it possible to initialize
indexes and retrievers over arbitrary modality combinations.
"""

from .base import OmniModalVectorStoreIndex
from .retriever import OmniModalVectorIndexRetriever

__all__ = ["OmniModalVectorStoreIndex", "OmniModalVectorIndexRetriever"]
