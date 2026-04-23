import base64
import re
import warnings
import logging
import io

from binascii import Error as BinasciiError
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import (
    Optional,
    List,
    Dict,
    Tuple,
    Callable,
    AsyncIterator,
    Awaitable,
    Any,
)
from urllib.parse import urlparse, parse_qs
from httpx import AsyncClient, Timeout
from mcp.client.session import ClientSession
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.shared.session import ProgressFnT
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import OAuthClientMetadata, OAuthToken, OAuthClientInformationFull
from mcp import types
from pydantic import AnyUrl

from llama_index.core.llms import (
    AudioBlock,
    ChatMessage,
    DocumentBlock,
    ImageBlock,
    TextBlock,
)


class StreamingHandler(logging.Handler):
    def __init__(
        self, callback: Callable, events: Optional[List[types.TextContent]]
    ) -> None:
        super().__init__()
        self.callback = callback
        self.events = events

    def emit(self, record) -> None:
        log_entry = self.format(record)
        self.callback(message=log_entry, events=self.events)  # Stream the message


def streaming_handler_callback(
    message: str, events: Optional[List[types.TextContent]]
) -> None:
    if not events:
        return
    events.append(types.TextContent(type="text", text=message))


def enable_sse(command_or_url: str) -> bool:
    """
    Check if the command or URL is an SSE endpoint.
    """
    url = urlparse(command_or_url)
    query_params = parse_qs(url.query)
    if "transport" in query_params and query_params["transport"][0] == "sse":
        return True
    elif url.path.rstrip("/").endswith("/sse"):
        return True
    elif "/sse/" in url.path:
        return True
    return False


class DefaultInMemoryTokenStorage(TokenStorage):
    """
    Simple in-memory token storage implementation for OAuth authentication.

    This is the default storage used when none is provided to with_oauth().
    Not suitable for production use across restarts as tokens are only stored
    in memory.
    """

    def __init__(self):
        self._tokens: Optional[OAuthToken] = None
        self._client_info: Optional[OAuthClientInformationFull] = None

    async def get_tokens(self) -> Optional[OAuthToken]:
        """Get the stored OAuth tokens."""
        return self._tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Store OAuth tokens."""
        self._tokens = tokens

    async def get_client_info(self) -> Optional[OAuthClientInformationFull]:
        """Get the stored client information."""
        return self._client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        """Store client information."""
        self._client_info = client_info


def _decode_mcp_base64(data: str) -> bytes:
    """
    Decode an MCP base64 payload robustly.
    """
    clean = re.sub(r"\s+", "", data)
    if not clean:
        raise BinasciiError("empty base64 payload")
    # Some non-spec servers embed a data URI prefix, so strip it.
    if clean.startswith("data:") and ";base64," in clean:
        clean = clean.split(";base64,", 1)[1]
        if not clean:
            raise BinasciiError("empty base64 payload after data URI prefix")
    clean = clean.replace("-", "+").replace("_", "/")
    pad = (-len(clean)) % 4
    clean += "=" * pad
    return base64.b64decode(clean, validate=True)


def _embedded_resource_to_blocks(content: types.EmbeddedResource) -> List:
    """
    Convert an MCP EmbeddedResource to a list of llama-index content blocks.
    TextResourceContents           -> TextBlock
    BlobResourceContents (image/*) -> ImageBlock
    BlobResourceContents (audio/*) -> AudioBlock
    BlobResourceContents (other)   -> DocumentBlock
    """
    res = content.resource

    if isinstance(res, types.TextResourceContents):
        uri = str(res.uri)
        return [TextBlock(text=f"[Embedded resource: {uri}]\n{res.text}")]

    if isinstance(res, types.BlobResourceContents):
        uri = str(res.uri)
        mime = res.mimeType.lower() if res.mimeType else None
        try:
            raw = _decode_mcp_base64(res.blob)
        except BinasciiError:
            warnings.warn(
                f"Failed to decode embedded blob resource ({uri}); rendering as text placeholder.",
                UserWarning,
                stacklevel=2,
            )
            return [TextBlock(text=f"[Binary resource: {uri}]")]

        if mime and mime.startswith("image/") and mime != "image/":
            return [ImageBlock(image=raw, image_mimetype=mime)]
        if mime and mime.startswith("audio/") and mime != "audio/":
            audio_format = mime.split("/", 1)[1].split(";", 1)[0].strip()
            return [AudioBlock(audio=raw, format=audio_format)]
        last_segment = uri.rstrip("/").rsplit("/", 1)[-1] or None
        return [
            DocumentBlock(
                data=raw,
                document_mimetype=mime,
                title=last_segment,
            )
        ]

    warnings.warn(
        f"Unknown EmbeddedResource.resource type: {type(res).__name__}; "
        "rendering as text placeholder.",
        UserWarning,
        stacklevel=2,
    )
    return [TextBlock(text="[Embedded resource: unsupported payload type]")]


def _content_to_blocks(content: types.ContentBlock) -> List:
    """
    Dispatch any MCP ContentBlock variant to llama-index content blocks.
    Handles all variants defined in the MCP spec ContentBlock:
    TextContent, ImageContent, AudioContent, ResourceLink, EmbeddedResource.
    Unknown future variants produce a warning and a TextBlock placeholder.
    """
    if isinstance(content, types.TextContent):
        return [TextBlock(text=content.text)]

    if isinstance(content, types.ImageContent):
        try:
            raw = _decode_mcp_base64(content.data)
            return [ImageBlock(image=raw, image_mimetype=content.mimeType)]
        except BinasciiError:
            warnings.warn(
                "Invalid base64 in ImageContent; rendering as placeholder.",
                UserWarning,
                stacklevel=2,
            )
            return [TextBlock(text="[ImageContent: invalid base64]")]

    if isinstance(content, types.AudioContent):
        try:
            raw = _decode_mcp_base64(content.data)
            audio_format = (
                content.mimeType.split("/", 1)[1].split(";", 1)[0].strip()
                if "/" in content.mimeType
                else None
            )
            return [AudioBlock(audio=raw, format=audio_format)]
        except BinasciiError:
            warnings.warn(
                "Invalid base64 in AudioContent; rendering as placeholder.",
                UserWarning,
                stacklevel=2,
            )
            return [TextBlock(text="[AudioContent: invalid base64]")]

    if isinstance(content, types.ResourceLink):
        display = content.title or content.name or str(content.uri)
        lines = [f"[ResourceLink] {display}", f"URI: {content.uri}"]
        if content.description:
            lines.append(content.description)
        return [TextBlock(text="\n".join(lines))]

    if isinstance(content, types.EmbeddedResource):
        return _embedded_resource_to_blocks(content)

    warnings.warn(
        f"Unsupported MCP content type: {type(content).__name__}; rendering placeholder.",
        UserWarning,
        stacklevel=2,
    )
    return [TextBlock(text=f"[Unsupported MCP content: {type(content).__name__}]")]


class BasicMCPClient(ClientSession):
    """
    Basic MCP client that can be used to connect to an MCP server.

    This is useful for connecting to any MCP server.

    Args:
        command_or_url: The command to run or the URL to connect to.
        args: The arguments to pass to StdioServerParameters.
        env: The environment variables to set for StdioServerParameters.
        timeout: The timeout for HTTP operations in seconds. Default is 30.
        sse_read_timeout: The timeout for SSE read operations in seconds. Default is 300 (5 minutes).
        auth: Optional OAuth client provider for authentication.
        sampling_callback: Optional callback for handling sampling messages.
        headers: Optional headers to pass by sse client or streamable http client
        tool_call_logs_callback: Async function to store the logs deriving from an MCP tool call: logs are provided as a list of strings, representing log messages. Defaults to None.
        http_client: Optional httpx AsyncClient to use for Streamable transport. Will ignore timeout and headers parameters if provided.

    """

    def __init__(
        self,
        command_or_url: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        sse_read_timeout: int = 300,
        auth: Optional[OAuthClientProvider] = None,
        sampling_callback: Optional[
            Callable[
                [types.CreateMessageRequestParams], Awaitable[types.CreateMessageResult]
            ]
        ] = None,
        headers: Optional[Dict[str, Any]] = None,
        tool_call_logs_callback: Optional[Callable[[List[str]], Awaitable[Any]]] = None,
        http_client: Optional[AsyncClient] = None,
    ):
        self.command_or_url = command_or_url
        self.args = args or []
        self.env = env or {}
        self.timeout = timeout
        self.sse_read_timeout = sse_read_timeout
        self.auth = auth
        self.sampling_callback = sampling_callback
        self.headers = headers
        self.tool_call_logs_callback = tool_call_logs_callback
        self.client_provided = http_client is not None
        self.http_client = (
            http_client
            if self.client_provided
            else create_mcp_http_client(
                timeout=Timeout(timeout, read=sse_read_timeout), headers=headers
            )
        )
        if auth is not None:
            self.http_client.auth = auth

    @classmethod
    def with_oauth(
        cls,
        command_or_url: str,
        client_name: str,
        redirect_uris: List[str],
        redirect_handler: Callable[[str], None],
        callback_handler: Callable[[], Tuple[str, Optional[str]]],
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        sse_read_timeout: int = 300,
        token_storage: Optional[TokenStorage] = None,
        tool_call_logs_callback: Optional[Callable[[List[str]], Awaitable[Any]]] = None,
        http_client: Optional[AsyncClient] = None,
    ) -> "BasicMCPClient":
        """
        Create a client with OAuth authentication.

        Args:
            command_or_url: The command to run or the URL to connect to
            client_name: The name of the OAuth client
            redirect_uris: The redirect URIs for the OAuth flow
            redirect_handler: Function that handles the redirect URL
            callback_handler: Function that returns the auth code and state
            token_storage: Optional token storage for OAuth client. If not provided,
                           a default in-memory storage is used (tokens will be lost on restart).
            args: The arguments to pass to StdioServerParameters.
            env: The environment variables to set for StdioServerParameters.
            timeout: The timeout for HTTP operations in seconds. Default is 30.
            sse_read_timeout: The timeout for SSE read operations in seconds. Default is 300.
            tool_call_logs_callback: Async function to store the logs deriving from an MCP tool call: logs are provided as a list of strings, representing log messages. Defaults to None.
            http_client: Optional httpx AsyncClient to use for Streamable transport. Will ignore timeout and headers parameters if provided.

        Returns:
            An authenticated MCP client

        """
        # Use default in-memory storage if none provided
        if token_storage is None:
            token_storage = DefaultInMemoryTokenStorage()
            warnings.warn(
                "Using default in-memory token storage. Tokens will be lost on restart.",
                UserWarning,
            )

        oauth_auth = OAuthClientProvider(
            server_url=command_or_url if urlparse(command_or_url).scheme else None,
            client_metadata=OAuthClientMetadata(
                client_name=client_name,
                redirect_uris=redirect_uris,
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
            ),
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
            storage=token_storage,
        )

        return cls(
            command_or_url,
            auth=oauth_auth,
            args=args,
            env=env,
            timeout=timeout,
            sse_read_timeout=sse_read_timeout,
            tool_call_logs_callback=tool_call_logs_callback,
            http_client=http_client,
        )

    @asynccontextmanager
    async def _run_session(self) -> AsyncIterator[ClientSession]:
        """Create and initialize a session with the MCP server."""
        url = urlparse(self.command_or_url)
        scheme = url.scheme

        if scheme in ("http", "https"):
            # Check if this is a streamable HTTP endpoint (default) or SSE
            if enable_sse(self.command_or_url):
                # SSE transport
                async with sse_client(
                    self.command_or_url,
                    auth=self.auth,
                    headers=self.headers,
                    timeout=self.timeout,
                    sse_read_timeout=self.sse_read_timeout,
                ) as streams:
                    async with ClientSession(
                        *streams,
                        read_timeout_seconds=timedelta(seconds=self.timeout),
                        sampling_callback=self.sampling_callback,
                    ) as session:
                        await session.initialize()
                        yield session
            else:
                # Streamable HTTP transport (recommended)
                async with streamable_http_client(
                    url=self.command_or_url,
                    http_client=self.http_client,
                ) as (read, write, _):
                    async with ClientSession(
                        read,
                        write,
                        read_timeout_seconds=timedelta(seconds=self.timeout),
                        sampling_callback=self.sampling_callback,
                    ) as session:
                        await session.initialize()
                        yield session
        else:
            # stdio transport
            server_parameters = StdioServerParameters(
                command=self.command_or_url, args=self.args, env=self.env
            )
            async with stdio_client(server_parameters) as streams:
                async with ClientSession(
                    *streams,
                    read_timeout_seconds=timedelta(seconds=self.timeout),
                    sampling_callback=self.sampling_callback,
                ) as session:
                    await session.initialize()
                    yield session

    def _configure_tool_call_logs_callback(self) -> io.StringIO:
        handler = io.StringIO()
        stream_handler = logging.StreamHandler(handler)

        # Configure logging to capture all events
        logging.basicConfig(
            level=logging.DEBUG,  # Capture all log levels
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s\n",
            handlers=[
                stream_handler,
            ],
        )
        # Also enable logging for specific FastMCP components
        fastmcp_logger = logging.getLogger("fastmcp")
        fastmcp_logger.setLevel(logging.DEBUG)

        # Enable HTTP transport logging to see network details
        http_logger = logging.getLogger("httpx")
        http_logger.setLevel(logging.DEBUG)

        return handler

    # Tool methods
    async def call_tool(
        self,
        tool_name: str,
        arguments: Optional[dict] = None,
        progress_callback: Optional[ProgressFnT] = None,
    ) -> types.CallToolResult:
        """Call a tool on the MCP server."""
        if self.tool_call_logs_callback is not None:
            # we use a string stream so that we can recover all logs at the end of the session
            handler = self._configure_tool_call_logs_callback()

            async with self._run_session() as session:
                result = await session.call_tool(
                    tool_name, arguments=arguments, progress_callback=progress_callback
                )

                # get all logs by dividing the string with \n, since the format of the log has an \n at the end of the log message
                extra_values = handler.getvalue().split("\n")

                # pipe the logs list into tool_call_logs_callback
                await self.tool_call_logs_callback(extra_values)

                return result
        else:
            async with self._run_session() as session:
                return await session.call_tool(
                    tool_name, arguments=arguments, progress_callback=progress_callback
                )

    async def list_tools(self) -> types.ListToolsResult:
        """List all available tools on the MCP server."""
        async with self._run_session() as session:
            return await session.list_tools()

    # Resource methods
    async def list_resources(self) -> types.ListToolsResult:
        """List all available resources on the MCP server."""
        async with self._run_session() as session:
            return await session.list_resources()

    async def list_resource_templates(self) -> types.ListToolsResult:
        """List all dynamic available resources on the MCP server."""
        async with self._run_session() as session:
            return await session.list_resource_templates()

    async def read_resource(self, resource_uri: AnyUrl) -> types.ReadResourceResult:
        """
        Read a resource from the MCP server.

        Returns:
            Tuple containing the resource content as bytes and the MIME type

        """
        async with self._run_session() as session:
            return await session.read_resource(resource_uri)

    ## ----- Prompt methods -----

    async def list_prompts(self) -> List[types.Prompt]:
        """List all available prompts on the MCP server."""
        async with self._run_session() as session:
            return await session.list_prompts()

    async def get_prompt(
        self, prompt_name: str, arguments: Optional[Dict[str, str]] = None
    ) -> List[ChatMessage]:
        """
        Get a prompt from the MCP server.

        Args:
            prompt_name: The name of the prompt to get
            arguments: Optional arguments to pass to the prompt

        Returns:
            The prompt as a list of llama-index ChatMessage objects

        """
        async with self._run_session() as session:
            prompt = await session.get_prompt(prompt_name, arguments)
            llama_messages = []
            for message in prompt.messages:
                blocks = _content_to_blocks(message.content)
                llama_messages.append(ChatMessage(role=message.role, blocks=blocks))
            return llama_messages
