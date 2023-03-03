"""Response schema."""

from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Union

from dataclasses_json import DataClassJsonMixin

from gpt_index.data_structs.data_structs import Node
from gpt_index.utils import truncate_text


@dataclass
class SourceNode(DataClassJsonMixin):
    """Source node.

    User-facing class containing the source text and the corresponding document id.

    """

    source_text: str
    doc_id: Optional[str]
    extra_info: Optional[Dict[str, Any]] = None
    node_info: Optional[Dict[str, Any]] = None

    # distance score between node and query, if applicable
    similarity: Optional[float] = None

    @classmethod
    def from_node(cls, node: Node, similarity: Optional[float] = None) -> "SourceNode":
        """Create a SourceNode from a Node."""
        return cls(
            source_text=node.get_text(),
            doc_id=node.ref_doc_id,
            extra_info=node.extra_info,
            node_info=node.node_info,
            similarity=similarity,
        )

    @classmethod
    def from_nodes(cls, nodes: List[Node]) -> List["SourceNode"]:
        """Create a list of SourceNodes from a list of Nodes."""
        return [cls.from_node(node) for node in nodes]


@dataclass
class Response:
    """Response object.

    Returned if streaming=False during the `index.query()` call.

    Attributes:
        response: The response text.

    """

    response: Optional[str]
    source_nodes: List[SourceNode] = field(default_factory=list)
    extra_info: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        """Convert to string representation."""
        return self.response or "None"

    def get_formatted_sources(self, length: int = 100) -> str:
        """Get formatted sources text."""
        texts = []
        for source_node in self.source_nodes:
            fmt_text_chunk = truncate_text(source_node.source_text, length)
            doc_id = source_node.doc_id or "None"
            source_text = f"> Source (Doc id: {doc_id}): {fmt_text_chunk}"
            texts.append(source_text)
        return "\n\n".join(texts)


@dataclass
class StreamingResponse:
    """StreamingResponse object.

    Returned if streaming=True during the `index.query()` call.

    Attributes:
        response_gen: The response generator.

    """

    response_gen: Optional[Generator]
    source_nodes: List[SourceNode] = field(default_factory=list)
    extra_info: Optional[Dict[str, Any]] = None
    response_txt: Optional[str] = None

    def __str__(self) -> str:
        """Convert to string representation."""
        if self.response_txt is None and self.response_gen is not None:
            response_txt = ""
            for text in self.response_gen:
                response_txt += text
            self.response_txt = response_txt
        return self.response_txt or "None"

    def get_response(self) -> Response:
        """Get a standard response object."""
        if self.response_txt is None and self.response_gen is not None:
            response_txt = ""
            for text in self.response_gen:
                response_txt += text
            self.response_txt = response_txt
        return Response(self.response_txt, self.source_nodes, self.extra_info)

    def print_response_stream(self) -> None:
        """Print the response stream."""
        if self.response_txt is None and self.response_gen is not None:
            response_txt = ""
            for text in self.response_gen:
                print(text, end="")
            self.response_txt = response_txt
        else:
            print(self.response_txt)

    def get_formatted_sources(self, length: int = 100) -> str:
        """Get formatted sources text."""
        texts = []
        for source_node in self.source_nodes:
            fmt_text_chunk = truncate_text(source_node.source_text, length)
            doc_id = source_node.doc_id or "None"
            source_text = f"> Source (Doc id: {doc_id}): {fmt_text_chunk}"
            texts.append(source_text)
        return "\n\n".join(texts)


RESPONSE_TYPE = Union[Response, StreamingResponse]
