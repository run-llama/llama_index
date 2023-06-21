"""Base schema for data structures."""
from abc import abstractmethod, abstractproperty
from dataclasses import dataclass
from enum import Enum, auto
from hashlib import sha256
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union

from llama_index.utils import get_new_id


DEFAULT_TEXT_NODE_TMPL = "{metadata_str}\n\n{content}"
DEFAULT_METADATA_TMPL = "{key}: {value}"


def _validate_is_flat_dict(metadata_dict: dict) -> None:
    """
    Validate that metadata dict is flat,
    and key is str, and value is one of (str, int, float, None).
    """
    for key, val in metadata_dict.items():
        if not isinstance(key, str):
            raise ValueError("Metadata key must be str!")
        if not isinstance(val, (str, int, float, type(None))):
            raise ValueError("Value must be one of (str, int, float, None)")


class NodeRelationship(str, Enum):
    """Node relationships used in `BaseNode` class.

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


class ObjectType(str, Enum):
    DOCUMENT = auto()
    TEXT = auto()
    IMAGE = auto()
    INDEX = auto()


class RelatedNodeInfo(BaseModel):
    node_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    hash: Optional[str] = None


RELATED_NODE_TYPE = Union[RelatedNodeInfo, List[RelatedNodeInfo]]


class BaseNode(BaseModel):
    """Base node Object.

    Generic abstract interface for retrievable nodes

    """

    _id: Optional[str] = Field(default=None, description="Unique ID of the node.")
    embedding: Optional[List[float]] = Field(
        default=None, description="Embedding of the node."
    )
    hash: Optional[str] = Field(default=None, description="Hash of the node content.")
    weight: float = Field(
        default=1.0,
        description="Optional field to weight the node similarity during retrieval.",
    )

    """"
    metadata fields
    - injected as part of the text shown to LLMs as context
    - used by vector DBs for metadata filtering

    This must be a flat dictionary, 
    and only uses str keys, and (str, int, float) values.
    """
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="A flat dictionary of metadata fields"
    )
    usable_metadata: Optional[List[str]] = Field(
        default=None,
        description="Metadata keys that are used during retrieval.",
    )
    relationships: Dict[NodeRelationship, RELATED_NODE_TYPE] = Field(
        default_factory=dict,
        description="A mapping of relationships to other node information.",
    )

    def __post_init__(self) -> None:
        """Post init."""
        # assign doc_id if not set
        if self._id is None:
            self._id = get_new_id(set())
        if self.hash is None:
            self.hash = self._generate_node_hash()

        if self.metadata is not None:
            _validate_is_flat_dict(self.metadata)

    @abstractmethod
    def _generate_node_hash(self) -> str:
        """Generate a hash to represent the node."""

    @classmethod
    @abstractmethod
    def get_type(cls) -> str:
        """Get Object type."""

    @abstractmethod
    def get_content(self) -> str:
        """Get object content."""

    @abstractmethod
    def metadata_str(self) -> str:
        """Extra info string."""

    @abstractproperty
    @property
    def is_content_none(self) -> bool:
        """Check if text is None."""

    @property
    def node_id(self) -> str:
        if self._id is None:
            raise ValueError("Node ID is not set!")
        return self._id

    @property
    def is_id_none(self) -> bool:
        """Check if doc_id is None."""
        return self.is_id_none is None

    @property
    def source_node(self) -> Optional[RelatedNodeInfo]:
        """Source object node.

        Extracted from the relationships field.

        """
        if NodeRelationship.SOURCE not in self.relationships:
            return None

        relation = self.relationships[NodeRelationship.SOURCE]
        if isinstance(relation, list): 
            raise ValueError("Source object must be a single RelatedNodeInfo object")
        return relation

    @property
    def prev_node(self) -> RelatedNodeInfo:
        """Prev node."""
        if NodeRelationship.PREVIOUS not in self.relationships:
            raise ValueError("Object does not have previous link")

        relation = self.relationships[NodeRelationship.PREVIOUS]
        if not isinstance(relation, RelatedNodeInfo):
            raise ValueError("Previous object must be a single RelatedNodeInfo object")
        return relation

    @property
    def next_node(self) -> RelatedNodeInfo:
        """Next node."""
        if NodeRelationship.NEXT not in self.relationships:
            raise ValueError("Object does not have next link")

        relation = self.relationships[NodeRelationship.NEXT]
        if not isinstance(relation, RelatedNodeInfo):
            raise ValueError("Next object must be a single RelatedNodeInfo object")
        return relation

    @property
    def parent_node(self) -> RelatedNodeInfo:
        """Parent node."""
        if NodeRelationship.PARENT not in self.relationships:
            raise ValueError("Object does not have parent link")

        relation = self.relationships[NodeRelationship.PARENT]
        if not isinstance(relation, RelatedNodeInfo):
            raise ValueError("Parent object must be a single RelatedNodeInfo object")
        return relation

    @property
    def child_nodes(self) -> List[RelatedNodeInfo]:
        """Child nodes."""
        if NodeRelationship.CHILD not in self.relationships:
            raise ValueError("Object does not have child objects")

        relation = self.relationships[NodeRelationship.PARENT]
        if not isinstance(relation, list):
            raise ValueError("Child objects must be a list of RelatedNodeInfo objects.")
        return relation

    @property
    def extra_info(self) -> Dict[str, Any]:
        """TODO: DEPRECATED: Extra info."""
        return self.metadata

    def get_embedding(self) -> List[float]:
        """Get embedding.

        Errors if embedding is None.

        """
        if self.embedding is None:
            raise ValueError("embedding not set.")
        return self.embedding


class TextNode(BaseNode):
    content: str
    start_char_idx: Optional[int] = Field(
        default=None, description="Start char index of the node."
    )
    end_char_idx: Optional[int] = Field(
        default=None, description="End char index of the node."
    )
    text_template: str = Field(
        default=DEFAULT_TEXT_NODE_TMPL,
        description=(
            "Template for how text is formatted, with {content} and "
            "{metadata_str} placeholders."
        ),
    )
    metadata_template: str = Field(
        default=DEFAULT_METADATA_TMPL,
        description=(
            "Template for how metadata is formatted, with {key} and "
            "{value} placeholders."
        ),
    )
    metadata_seperator: str = Field(
        default="\n",
        description="Seperator between metadata fields when converting to string.",
    )

    def _generate_node_hash(self) -> str:
        """Generate a hash to represent the node."""
        doc_identity = str(self.content) + str(self.metadata_str)
        return sha256(doc_identity.encode("utf-8", "surrogatepass")).hexdigest()

    @classmethod
    def get_type(cls) -> str:
        """Get Object type."""
        return ObjectType.TEXT

    @property
    def is_content_none(self) -> bool:
        """Check if content is None."""
        return self.content == ""

    def get_content(self) -> str:
        """Get object content."""
        metadata_str = self.metadata_str()
        return self.text_template.format(
            content=self.content, metadata_str=metadata_str
        )

    def metadata_str(self) -> str:
        """Convert metadata to a string."""

        return self.metadata_seperator.join(
            [
                self.metadata_template.format(key=key, value=str(value))
                for key, value in self.metadata.items()
                if self.usable_metadata is None or key in self.usable_metadata
            ]
        )

    def get_node_info(self) -> Dict[str, Any]:
        """Get node info."""
        return {"start": self.start_char_idx, "end": self.end_char_idx}

    # TODO: deprecated node properties
    def get_text(self) -> str:
        """Deprecated: Get text."""
        return self.get_content()

    @property
    def ref_doc_id(self) -> Optional[RELATED_NODE_TYPE]:
        """Deprecated: Get ref doc id."""
        return self.source_node

    @property
    def node_info(self) -> Dict[str, Any]:
        """Deprecated: Get node info."""
        return self.get_node_info()


class ImageNode(TextNode):
    """Node with image."""

    # TODO: store reference instead of actual image
    # base64 encoded image str
    image: str

    @classmethod
    def get_type(cls) -> str:
        return ObjectType.IMAGE


@dataclass
class IndexNode(TextNode):
    """Node with reference to an index."""

    index_id: str

    @classmethod
    def get_type(cls) -> str:
        return ObjectType.INDEX


class NodeWithScore(BaseModel):
    node: BaseNode
    score: Optional[float] = None
