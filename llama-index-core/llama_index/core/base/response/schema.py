"""Response schema."""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.schema import NodeWithScore
from llama_index.core.types import TokenGen, TokenAsyncGen
from llama_index.core.utils import truncate_text


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


@dataclass
class AsyncStreamingResponse:
    """AsyncStreamingResponse object.

    Returned if streaming=True while using async.

    Attributes:
        async_response_gen: The response async generator.

    """

    async_response_gen: TokenAsyncGen
    source_nodes: List[NodeWithScore] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    response_txt: Optional[str] = None

    def __post_init__(self) -> None:
        self._lock = asyncio.Lock()

    def __str__(self) -> str:
        """Convert to string representation."""
        return asyncio.run(self._async_str)

    async def _async_str(self) -> str:
        """Convert to string representation."""
        async for _ in self._yield_response():
            ...
        return self.response_txt or "None"

    async def _yield_response(self) -> TokenAsyncGen:
        """Yield the string response."""
        async with self._lock:
            if self.response_txt is None and self.async_response_gen is not None:
                self.response_txt = ""
                async for text in self.async_response_gen:
                    self.response_txt += text
                    yield text
            else:
                yield self.response_txt

    async def async_response_gen(self) -> TokenAsyncGen:
        """Yield the string response."""
        async for text in self._yield_response():
            yield text

    async def get_response(self) -> Response:
        """Get a standard response object."""
        async for _ in self._yield_response():
            ...
        return Response(self.response_txt, self.source_nodes, self.metadata)

    async def print_response_stream(self) -> None:
        """Print the response stream."""
        streaming = True
        async for text in self._yield_response():
            print(text, end="", flush=True)
        # do an empty print to print on the next line again next time
        print()

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


RESPONSE_TYPE = Union[
    Response, StreamingResponse, AsyncStreamingResponse, PydanticResponse
]
