"""Response schema."""

from dataclasses import dataclass, field
from typing import List, Optional

from gpt_index.data_structs.data_structs import Node
from gpt_index.indices.utils import truncate_text


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

    def __str__(self) -> str:
        """Convert to string representation."""
        return self.response or "None"

    def get_formatted_sources(self) -> str:
        """Get formatted sources text."""
        texts = []
        for source_node in self.source_nodes:
            fmt_text_chunk = truncate_text(source_node.source_text, 100)
            doc_id = source_node.doc_id or "None"
            texts.append(f">Source (Doc id: {doc_id}): {fmt_text_chunk}")
        return "\n\n".join(texts)
