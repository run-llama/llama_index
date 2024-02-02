"""Response schema."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from llama_index.legacy.bridge.pydantic import BaseModel
from llama_index.legacy.schema import NodeWithScore
from llama_index.legacy.types import TokenGen
from llama_index.legacy.utils import truncate_text


@dataclass
class Response:
    """Response object.

    Returned if streaming=False.

    Attributes:
        response: The response text.

    """

    response: Optional[str]
    source_nodes: List[NodeWithScore] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        """Convert to string representation."""
        return self.response or "None"

    def get_formatted_sources(self, length: int = 100) -> str:
        """Get formatted sources text."""
        texts = []
        for source_node in self.source_nodes:
            fmt_text_chunk = truncate_text(source_node.node.get_content(), length)
            doc_id = source_node.node.node_id or "None"
            source_text = f"> Source (Doc id: {doc_id}): {fmt_text_chunk}"
            texts.append(source_text)
        return "\n\n".join(texts)


@dataclass
class PydanticResponse:
    """PydanticResponse object.

    Returned if streaming=False.

    Attributes:
        response: The response text.

    """

    response: Optional[BaseModel]
    source_nodes: List[NodeWithScore] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        """Convert to string representation."""
        return self.response.json() if self.response else "None"

    def __getattr__(self, name: str) -> Any:
        """Get attribute, but prioritize the pydantic  response object."""
        if self.response is not None and name in self.response.dict():
            return getattr(self.response, name)
        else:
            return None

    def get_formatted_sources(self, length: int = 100) -> str:
        """Get formatted sources text."""
        texts = []
        for source_node in self.source_nodes:
            fmt_text_chunk = truncate_text(source_node.node.get_content(), length)
            doc_id = source_node.node.node_id or "None"
            source_text = f"> Source (Doc id: {doc_id}): {fmt_text_chunk}"
            texts.append(source_text)
        return "\n\n".join(texts)

    def get_response(self) -> Response:
        """Get a standard response object."""
        response_txt = self.response.json() if self.response else "None"
        return Response(response_txt, self.source_nodes, self.metadata)


@dataclass
class StreamingResponse:
    """StreamingResponse object.

    Returned if streaming=True.

    Attributes:
        response_gen: The response generator.

    """

    response_gen: TokenGen
    source_nodes: List[NodeWithScore] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
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
        return Response(self.response_txt, self.source_nodes, self.metadata)

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

    def get_formatted_sources(self, length: int = 100, trim_text: int = True) -> str:
        """Get formatted sources text."""
        texts = []
        for source_node in self.source_nodes:
            fmt_text_chunk = source_node.node.get_content()
            if trim_text:
                fmt_text_chunk = truncate_text(fmt_text_chunk, length)
            node_id = source_node.node.node_id or "None"
            source_text = f"> Source (Node id: {node_id}): {fmt_text_chunk}"
            texts.append(source_text)
        return "\n\n".join(texts)


RESPONSE_TYPE = Union[Response, StreamingResponse, PydanticResponse]
