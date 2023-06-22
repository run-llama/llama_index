"""Node parser interface."""
from typing import List, Sequence

from abc import ABC, abstractmethod

from llama_index.data_structs.node import Node
from llama_index.readers.schema.base import Document


class NodeParser(ABC):
    """Base interface for node parser."""

    @abstractmethod
    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
    ) -> List[Node]:
        """Parse documents into nodes.

        Args:
            documents (Sequence[Document]): documents to parse

        """
