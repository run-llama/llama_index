"""Base schema for data structures."""
import warnings
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from hashlib import sha256
from typing import Any, Dict, List, Optional

from dataclasses_json import DataClassJsonMixin

from llama_index.utils import get_new_id


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


class DataRelationship(str, Enum):
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


class ObjectType(str, Enum):
    DOCUMENT = auto()
    TEXT = auto()
    IMAGE = auto()
    INDEX = auto()


@dataclass
class BaseDocument(DataClassJsonMixin):
    """Base document.

    Generic abstract interfaces that captures both index structs
    as well as documents.

    """

    # TODO: consolidate fields from Document/IndexStruct into base class
    text: Optional[str] = None
    doc_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    doc_hash: Optional[str] = None

    """"
    metadata fields
    - injected as part of the text shown to LLMs as context
    - used by vector DBs for metadata filtering

    This must be a flat dictionary, 
    and only uses str keys, and (str, int, float) values.
    """
    extra_info: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Post init."""
        # assign doc_id if not set
        if self.doc_id is None:
            self.doc_id = get_new_id(set())
        if self.doc_hash is None:
            self.doc_hash = self._generate_doc_hash()

        if self.extra_info is not None:
            _validate_is_flat_dict(self.extra_info)

    def _generate_doc_hash(self) -> str:
        """Generate a hash to represent the document."""
        doc_identity = str(self.text) + str(self.extra_info)
        return sha256(doc_identity.encode("utf-8", "surrogatepass")).hexdigest()

    @classmethod
    @abstractmethod
    def get_type(cls) -> str:
        """Get Document type."""

    @classmethod
    def get_types(cls) -> List[str]:
        """Get Document type."""
        # TODO: remove this method
        # a hack to preserve backwards compatibility for vector indices
        return [cls.get_type()]

    def get_text(self) -> str:
        """Get text."""
        if self.text is None:
            raise ValueError("text field not set.")
        return self.text

    def get_doc_id(self) -> str:
        """Get doc_id."""
        if self.doc_id is None:
            raise ValueError("doc_id not set.")
        return self.doc_id

    def get_doc_hash(self) -> str:
        """Get doc_hash."""
        if self.doc_hash is None:
            raise ValueError("doc_hash is not set.")
        return self.doc_hash

    @property
    def is_doc_id_none(self) -> bool:
        """Check if doc_id is None."""
        return self.doc_id is None

    @property
    def is_text_none(self) -> bool:
        """Check if text is None."""
        return self.text is None

    def get_embedding(self) -> List[float]:
        """Get embedding.

        Errors if embedding is None.

        """
        if self.embedding is None:
            raise ValueError("embedding not set.")
        return self.embedding

    @property
    def extra_info_str(self) -> Optional[str]:
        """Extra info string."""
        if self.extra_info is None:
            return None

        return "\n".join([f"{k}: {str(v)}" for k, v in self.extra_info.items()])


@dataclass
class BaseIndexObject(DataClassJsonMixin):
    """Base Index Object.

    Generic abstract interface for retrievable objects

    """

    _id: str = ""
    content: Any = None
    relationships: Dict[DataRelationship, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    object_hash: Optional[str] = None
    weight: float = 1.0

    """"
    metadata fields
    - injected as part of the text shown to LLMs as context
    - used by vector DBs for metadata filtering

    This must be a flat dictionary, 
    and only uses str keys, and (str, int, float) values.
    """
    metadata: Dict[str, Any] = field(default_factory=dict)

    # attributes are additional fields that do not influence query/retrieval
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Post init."""
        # assign doc_id if not set
        if not self._id:
            self._id = get_new_id(set())
        if self.object_hash is None:
            self.object_hash = self._generate_object_hash()

        if self.metadata is not None:
            _validate_is_flat_dict(self.metadata)

    @abstractmethod
    def _generate_object_hash(self) -> str:
        """Generate a hash to represent the object."""

    @classmethod
    @abstractmethod
    def get_type(cls) -> str:
        """Get Object type."""

    @abstractmethod
    def get_content(self) -> str:
        """Get object content."""

    def get_object_id(self) -> str:
        """Get doc_id."""
        if self._id is None:
            raise ValueError("_id not set.")
        return self._id

    def get_object_hash(self) -> str:
        """Get object_hash."""
        if self.object_hash is None:
            raise ValueError("object_hash is not set.")
        return self.object_hash

    @property
    def object_id(self) -> str:
        return self._id

    @property
    def is_id_none(self) -> bool:
        """Check if doc_id is None."""
        return self.is_id_none is None

    @property
    def is_content_none(self) -> bool:
        """Check if text is None."""
        return self.content is None

    def get_embedding(self) -> List[float]:
        """Get embedding.

        Errors if embedding is None.

        """
        if self.embedding is None:
            raise ValueError("embedding not set.")
        return self.embedding

    @property
    def metadata_str(self) -> str:
        """Extra info string."""
        if self.metadata is None:
            return ""

        return "\n".join([f"{k}: {str(v)}" for k, v in self.metadata.items()])

    @property
    def source_object_id(self) -> Optional[str]:
        """Source object id.

        Extracted from the relationships field.

        """
        return self.relationships.get(DataRelationship.SOURCE, None)

    @property
    def prev_object_id(self) -> str:
        """Prev object id."""
        if DataRelationship.PREVIOUS not in self.relationships:
            raise ValueError("Object does not have previous link")
        if not isinstance(self.relationships[DataRelationship.PREVIOUS], str):
            raise ValueError("Previous object must be a string id")
        return self.relationships[DataRelationship.PREVIOUS]

    @property
    def next_object_id(self) -> str:
        """Next object id."""
        if DataRelationship.NEXT not in self.relationships:
            raise ValueError("Object does not have next link")
        if not isinstance(self.relationships[DataRelationship.NEXT], str):
            raise ValueError("Next object must be a string id")
        return self.relationships[DataRelationship.NEXT]

    @property
    def parent_object_id(self) -> str:
        """Parent object id."""
        if DataRelationship.PARENT not in self.relationships:
            raise ValueError("Object does not have parent link")
        if not isinstance(self.relationships[DataRelationship.PARENT], str):
            raise ValueError("Parent object must be a string id")
        return self.relationships[DataRelationship.PARENT]

    @property
    def child_object_ids(self) -> List[str]:
        """Child object ids."""
        if DataRelationship.CHILD not in self.relationships:
            raise ValueError("Object does not have child objects")
        if not isinstance(self.relationships[DataRelationship.CHILD], list):
            raise ValueError("Child objects must be a list ids")
        return self.relationships[DataRelationship.CHILD]

    @property
    def extra_info(self) -> Dict[str, Any]:
        return self.metadata


@dataclass
class BaseNode(BaseIndexObject):
    content: str = ""

    def _generate_object_hash(self) -> str:
        """Generate a hash to represent the object."""
        doc_identity = str(self.content) + str(self.metadata_str)
        return sha256(doc_identity.encode("utf-8", "surrogatepass")).hexdigest()

    @classmethod
    def get_type(cls) -> str:
        """Get Object type."""
        return ObjectType.TEXT

    def get_content(self) -> str:
        """Get object content."""
        return (f"{self.metadata_str}\n\n" f"{self.content}").strip()

    def get_node_info(self) -> Dict[str, Any]:
        """DEPRECATED: Get node info."""
        if self.attributes is None:
            raise ValueError("Node attributes not set.")
        return self.attributes

    def get_text(self) -> str:
        """DEPRECATED: Get text."""
        return self.get_content()

    def get_origin_type(self) -> str:
        """Get origin type."""
        if self.attributes is None or "_node_type" not in self.attributes:
            return self.get_type()
        return self.attributes["_node_type"]

    # TODO: deprecated node properties
    @property
    def ref_doc_id(self) -> Optional[str]:
        return self.source_object_id

    @property
    def prev_node_id(self) -> str:
        return self.prev_object_id

    @property
    def next_node_id(self) -> str:
        return self.next_object_id

    @property
    def parent_node_id(self) -> str:
        return self.parent_object_id

    @property
    def child_node_ids(self) -> List[str]:
        return self.child_object_ids

    @property
    def node_info(self) -> Dict[str, Any]:
        return self.attributes


@dataclass
class Document(BaseNode):
    title: Optional[str] = None
    description: Optional[str] = None

    def _generate_object_hash(self) -> str:
        """Generate a hash to represent the object."""
        doc_identity = str(self.title) + str(self.description) + str(self.metadata_str)
        return sha256(doc_identity.encode("utf-8", "surrogatepass")).hexdigest()

    @classmethod
    def get_type(cls) -> str:
        """Get Object type."""
        return ObjectType.DOCUMENT

    def get_content(self) -> str:
        """Get object content."""
        return (
            f"Title: {self.title}\n"
            f"Description: {self.description}\n"
            f"{self.metadata_str}"
        ).strip()


class ImageNode(BaseNode):
    """Node with image."""

    # TODO: store reference instead of actual image
    # base64 encoded image str

    @classmethod
    def get_type(cls) -> str:
        return ObjectType.IMAGE

    @property
    def image(self) -> str:
        return self.content


@dataclass
class IndexNode(BaseNode):
    """Node with reference to an index."""

    @classmethod
    def get_type(cls) -> str:
        return ObjectType.INDEX

    @property
    def index_id(self) -> str:
        return self.content


@dataclass
class NodeWithScore(DataClassJsonMixin):
    node: BaseNode
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
