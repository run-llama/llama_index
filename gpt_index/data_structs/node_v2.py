from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from gpt_index.schema import BaseDocument


class DocumentRelationship(Enum):
    SOURCE = auto()
    PREVIOUS = auto()
    NEXT = auto()


@dataclass
class Node(BaseDocument):
    """A generic node of data.

    Base struct used in most indices.

    """

    def __post_init__(self) -> None:
        """Post init."""
        super().__post_init__()
        # NOTE: for Node objects, the text field is required
        if self.text is None:
            raise ValueError("text field not set.")

    # embeddings
    embedding: Optional[List[float]] = None

    # extra node info
    node_info: Optional[Dict[str, Any]] = None

    # TODO: store reference instead of actual image
    # base64 encoded image str
    image: Optional[str] = None

    # document relationships
    relationships: Dict[DocumentRelationship, str] = field(default_factory=dict)

    @property
    def ref_doc_id(self) -> str:
        """reference document id."""
        if DocumentRelationship.SOURCE not in self.relationships:
            raise ValueError("Node does not have source doc")
        return self.relationships[DocumentRelationship.SOURCE]

    @property
    def prev_node_id(self) -> str:
        if DocumentRelationship.PREVIOUS not in self.relationships:
            raise ValueError("Node does not have previous node")
        return self.relationships[DocumentRelationship.PREVIOUS]

    @property
    def next_node_id(self) -> str:
        if DocumentRelationship.NEXT not in self.relationships:
            raise ValueError("Node does not have next node")
        return self.relationships[DocumentRelationship.NEXT]

    def get_text(self) -> str:
        """Get text."""
        text = super().get_text()
        result_text = (
            text if self.extra_info_str is None else f"{self.extra_info_str}\n\n{text}"
        )
        return result_text

    @classmethod
    def get_type(cls) -> str:
        """Get type."""
        # TODO: consolidate with IndexStructType
        return "node"
