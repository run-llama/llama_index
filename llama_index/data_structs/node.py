"""`Node` data structure.

`Node` is a generic data container that contains
a piece of data (e.g. chunk of text, an image, a table, etc).

In comparison to a raw `Document`, it contains additional metadata
about its relationship to other `Node` objects (and `Document` objects).

It is often used as an atomic unit of data in various indices.
"""
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from dataclasses_json import DataClassJsonMixin

from llama_index.schema import BaseDocument

_logger = logging.getLogger(__name__)


class DocumentRelationship(str, Enum):
    """Document relationships used in `Node` class.

    Attributes:
        SOURCE: The node is the source document.
        PREVIOUS: The node is the previous node in the document.
        NEXT: The node is the next node in the document.
        PARENT: The node is the parent node in the document.
        CHILD: The node is a child node in the document.

    """

    SOURCE = auto()
    PREVIOUS = auto()
    NEXT = auto()
    PARENT = auto()
    CHILD = auto()


class NodeType(str, Enum):
    TEXT = auto()
    IMAGE = auto()
    INDEX = auto()


@dataclass
class Node(BaseDocument):
    """A generic node of data.

    Arguments:
        text (str): The text of the node.
        doc_id (Optional[str]): The document id of the node.
        embeddings (Optional[List[float]]): The embeddings of the node.
        relationships (Dict[DocumentRelationship, Any]): The relationships of the node.

    """

    def __post_init__(self) -> None:
        """Post init."""
        super().__post_init__()
        # NOTE: for Node objects, the text field is required
        if self.text is None:
            raise ValueError("text field not set.")

        if self.node_info is None:
            self.node_info = {}
        if "_node_type" not in self.node_info:
            self.node_info["_node_type"] = self.get_type()

    # extra node info
    node_info: Optional[Dict[str, Any]] = None

    # document relationships
    relationships: Dict[DocumentRelationship, Any] = field(default_factory=dict)

    def get_node_info(self) -> Dict[str, Any]:
        """Get node info."""
        if self.node_info is None:
            raise ValueError("Node info not set.")
        return self.node_info

    @property
    def ref_doc_id(self) -> Optional[str]:
        """Source document id.

        Extracted from the relationships field.

        """
        return self.relationships.get(DocumentRelationship.SOURCE, None)

    @property
    def prev_node_id(self) -> str:
        """Prev node id."""
        if DocumentRelationship.PREVIOUS not in self.relationships:
            raise ValueError("Node does not have previous node")
        if not isinstance(self.relationships[DocumentRelationship.PREVIOUS], str):
            raise ValueError("Previous node must be a string")
        return self.relationships[DocumentRelationship.PREVIOUS]

    @property
    def next_node_id(self) -> str:
        """Next node id."""
        if DocumentRelationship.NEXT not in self.relationships:
            raise ValueError("Node does not have next node")
        if not isinstance(self.relationships[DocumentRelationship.NEXT], str):
            raise ValueError("Next node must be a string")
        return self.relationships[DocumentRelationship.NEXT]

    @property
    def parent_node_id(self) -> str:
        """Parent node id."""
        if DocumentRelationship.PARENT not in self.relationships:
            raise ValueError("Node does not have parent node")
        if not isinstance(self.relationships[DocumentRelationship.PARENT], str):
            raise ValueError("Parent node must be a string")
        return self.relationships[DocumentRelationship.PARENT]

    @property
    def child_node_ids(self) -> List[str]:
        """Child node ids."""
        if DocumentRelationship.CHILD not in self.relationships:
            raise ValueError("Node does not have child nodes")
        if not isinstance(self.relationships[DocumentRelationship.CHILD], list):
            raise ValueError("Child nodes must be a list")
        return self.relationships[DocumentRelationship.CHILD]

    def get_text(self) -> str:
        """Get text."""
        text = super().get_text()
        extra_info_exists = self.extra_info is not None and len(self.extra_info) > 0
        result_text = (
            text if not extra_info_exists else f"{self.extra_info_str}\n\n{text}"
        )
        return result_text

    @classmethod
    def get_type(cls) -> str:
        """Get type."""
        return NodeType.TEXT

    def get_origin_type(self) -> str:
        """Get origin type."""
        if self.node_info is None or "_node_type" not in self.node_info:
            return self.get_type()
        return self.node_info["_node_type"]


@dataclass
class ImageNode(Node):
    """Node with image."""

    # TODO: store reference instead of actual image
    # base64 encoded image str
    image: Optional[str] = None

    @classmethod
    def get_type(cls) -> str:
        return NodeType.IMAGE


@dataclass
class IndexNode(Node):
    """Node with reference to an index."""

    index_id: Optional[str] = None

    @classmethod
    def get_type(cls) -> str:
        return NodeType.INDEX


@dataclass
class NodeWithScore(DataClassJsonMixin):
    node: Node
    score: Optional[float] = None

    @property
    def doc_id(self) -> Optional[str]:
        warnings.warn(".doc_id is deprecated, use .node.ref_doc_id instead")
        return self.node.ref_doc_id

    @property
    def source_text(self) -> str:
        warnings.warn(".source_text is deprecated, use .node.get_text() instead")
        return self.node.get_text()

    @property
    def extra_info(self) -> Optional[Dict[str, Any]]:
        warnings.warn(".extra_info is deprecated, use .node.extra_info instead")
        return self.node.extra_info

    @property
    def node_info(self) -> Optional[Dict[str, Any]]:
        warnings.warn(".node_info is deprecated, use .node.node_info instead")
        return self.node.node_info

    @property
    def similarity(self) -> Optional[float]:
        warnings.warn(".similarity is deprecated, use .score instead instead")
        return self.score

    @property
    def image(self) -> Optional[str]:
        warnings.warn(
            ".image is deprecated, check if Node is an ImageNode \
            and use .node.image instead"
        )
        if isinstance(self.node, ImageNode):
            return self.node.image
        else:
            return None
