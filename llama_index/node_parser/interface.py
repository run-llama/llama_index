"""Node parser interface."""
from abc import ABC, abstractmethod
from typing import Any, List, Sequence

from llama_index.schema import TransformComponent, BaseNode, Document


class NodeParser(TransformComponent, ABC):
    """Base interface for node parser."""

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Parse documents into nodes.

        Args:
            documents (Sequence[Document]): documents to parse

        """

    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        return self.get_nodes_from_documents(nodes, **kwargs)
