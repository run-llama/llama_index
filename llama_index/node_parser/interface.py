"""Node parser interface."""
from typing import List, Sequence, Dict, Optional, Any

from abc import ABC, abstractmethod

from llama_index.schema import Document
from llama_index.schema import BaseNode


class NodeParser(ABC):
    """Base interface for node parser."""

    @abstractmethod
    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Parse documents into nodes.

        Args:
            documents (Sequence[Document]): documents to parse

        """


class BaseExtractor(ABC):
    """Base interface for feature extractor."""

    @abstractmethod
    def extract(
        self,
        nodes: List[BaseNode],
    ) -> List[Dict]:
        """Post process nodes parsed from documents.

        Args:
            nodes (List[BaseNode]): nodes to extract from
        """
