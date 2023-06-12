"""Response schema."""

from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Union

from llama_index.data_structs.node import NodeWithScore
from llama_index.utils import truncate_text


@dataclass
class Response:
    """Response object.

    Returned if streaming=False.

    Attributes:
        response: The response text.

    """

    response: Optional[str]
    source_nodes: List[NodeWithScore] = field(default_factory=list)
    extra_info: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        """Convert to string representation."""
        return self.response or "None"

    def get_formatted_sources(self, length: int = 100) -> str:
        """Get formatted sources text."""
        texts = []
        for source_node in self.source_nodes:
            fmt_text_chunk = truncate_text(source_node.node.get_text(), length)
            doc_id = source_node.node.doc_id or "None"
            source_text = f"> Source (Doc id: {doc_id}): {fmt_text_chunk}"
            texts.append(source_text)
        return "\n\n".join(texts)
    
    def formatted_source_nodes(self) -> str:
        """Get formatted source nodes as a string."""
        formatted_sources = []
        for node_with_score in self.source_nodes:
            truncated_text = truncate_text(node_with_score.node.text or "", 100)
            formatted_sources.append("NodeWithScore(")
            formatted_sources.append("  node=Node(")
            formatted_sources.append("    text='{}',".format(truncated_text.replace('\n', '\\n')))
            formatted_sources.append("    embedding={}, ".format(node_with_score.node.embedding))
            formatted_sources.append("    doc_hash='{}', ".format(node_with_score.node.doc_hash))
            formatted_sources.append("    extra_info={}, ".format(node_with_score.node.extra_info))
            formatted_sources.append("    node_info={}), ".format(node_with_score.node.node_info))
            formatted_sources.append("  relationships={")
            for relationship, value in node_with_score.node.relationships.items():
                formatted_sources.append("    {}: '{}',".format(relationship, value))
            formatted_sources.append("  }),")
            formatted_sources.append("  score={}".format(node_with_score.score))
            formatted_sources.append(")")
            formatted_sources.append("\n")
        return "\n".join(formatted_sources)


@dataclass
class StreamingResponse:
    """StreamingResponse object.

    Returned if streaming=True.

    Attributes:
        response_gen: The response generator.

    """

    response_gen: Optional[Generator]
    source_nodes: List[NodeWithScore] = field(default_factory=list)
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
                print(text, end="", flush=True)
                response_txt += text
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
