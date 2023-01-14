"""Response schema."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from gpt_index.data_structs.data_structs import Node
from gpt_index.indices.utils import truncate_text
from gpt_index.readers.schema.base import Document


@dataclass
class SourceNode:
    """Source node.

    User-facing class containing the source text and the corresponding document id.

    """

    source_text: str
    doc_id: Optional[str]

    @classmethod
    def from_node(cls, node: Node) -> "SourceNode":
        """Create a SourceNode from a Node."""
        return cls(source_text=node.get_text(), doc_id=node.ref_doc_id)

    @classmethod
    def from_nodes(cls, nodes: List[Node]) -> List["SourceNode"]:
        """Create a list of SourceNodes from a list of Nodes."""
        return [cls.from_node(node) for node in nodes]


@dataclass
class Response:
    """Response.

    Attributes:
        response: The response text.

    """

    response: Optional[str]
    source_nodes: List[SourceNode] = field(default_factory=list)
    extra_info: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        """Convert to string representation."""
        return self.response or "None"

    def get_formatted_sources(self, documents: Optional[List[Document]] = None) -> str:
        """Get formatted sources text."""
        texts = []
        for source_node in self.source_nodes:
            fmt_text_chunk = truncate_text(source_node.source_text, 100)
            doc_id = source_node.doc_id or "None"
            source_text = f"> Source (Doc id: {doc_id}): {fmt_text_chunk}"

            # If `documents` exists, add some helpful info about the document
            if documents is not None:
                source_document = next(
                    (d for d in documents if d.doc_id == doc_id), None
                )
                if (
                    source_document is not None
                    and source_document.extra_info is not None
                ):
                    source_text += f"\nMetadata: {source_document.extra_info}"
            texts.append(source_text)
        return "\n\n".join(texts)
