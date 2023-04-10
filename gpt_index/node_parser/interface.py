"""Node parser interface."""
from typing import List, Sequence

from abc import ABC, abstractmethod

from gpt_index.data_structs.node_v2 import Node
from gpt_index.readers.schema.base import Document


class NodeParser(ABC):
    """Base interface for node parser."""

    @abstractmethod
    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
    ) -> List[Node]:
        """Parse documents into nodes.

        Args:
            documents (Sequence[Document]): documents to parse

        """
