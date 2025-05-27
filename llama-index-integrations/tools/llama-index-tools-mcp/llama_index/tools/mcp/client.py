import warnings
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Optional, List, Dict, Tuple, Callable, AsyncIterator, Awaitable, Dict
from urllib.parse import urlparse

from mcp.client.session import ClientSession, ProgressFnT
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import OAuthClientMetadata, OAuthToken, OAuthClientInformationFull
from mcp import types

from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock


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


class BasicMCPClient(ClientSession):
    """
    Basic MCP client that can be used to connect to an MCP server.

    This is useful for connecting to any MCP server.

    Args:
        command_or_url: The command to run or the URL to connect to.
        args: The arguments to pass to StdioServerParameters.
        env: The environment variables to set for StdioServerParameters.
        timeout: The timeout for the command in seconds.
        auth: Optional OAuth client provider for authentication.
        sampling_callback: Optional callback for handling sampling messages.

    """

    def __init__(
        self,
        command_or_url: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        auth: Optional[OAuthClientProvider] = None,
        sampling_callback: Optional[
            Callable[
                [types.CreateMessageRequestParams], Awaitable[types.CreateMessageResult]
            ]
        ] = None,
    ):
        self.command_or_url = command_or_url
        self.args = args or []
        self.env = env or {}
        self.timeout = timeout
        self.auth = auth
        self.sampling_callback = sampling_callback

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
        token_storage: Optional[TokenStorage] = None,
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
            timeout: The timeout for the command in seconds.

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

        return cls(command_or_url, auth=oauth_auth, args=args, env=env, timeout=timeout)

    @asynccontextmanager
    async def _run_session(self) -> AsyncIterator[ClientSession]:
        """Create and initialize a session with the MCP server."""
        url = urlparse(self.command_or_url)
        scheme = url.scheme

        if scheme in ("http", "https"):
            # Check if this is a streamable HTTP endpoint (default) or SSE
            if url.path.endswith("/sse"):
                # SSE transport
                async with sse_client(self.command_or_url, auth=self.auth) as streams:
                    async with ClientSession(
                        *streams,
                        read_timeout_seconds=timedelta(seconds=self.timeout),
                        sampling_callback=self.sampling_callback,
                    ) as session:
                        await session.initialize()
                        yield session
            else:
                # Streamable HTTP transport (recommended)
                async with streamablehttp_client(
                    self.command_or_url, auth=self.auth
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

    # Tool methods
    async def call_tool(
        self,
        tool_name: str,
        arguments: Optional[dict] = None,
        progress_callback: Optional[ProgressFnT] = None,
    ) -> types.CallToolResult:
        """Call a tool on the MCP server."""
        async with self._run_session() as session:
            return await session.call_tool(
                tool_name, arguments=arguments, progress_callback=progress_callback
            )

    async def list_tools(self) -> types.ListToolsResult:
        """List all available tools on the MCP server."""
        async with self._run_session() as session:
            return await session.list_tools()

    # Resource methods
    async def list_resources(self) -> types.ListToolsRequest:
        """List all available resources on the MCP server."""
        async with self._run_session() as session:
            return await session.list_resources()

    async def read_resource(self, resource_name: str) -> types.ReadResourceResult:
        """
        Read a resource from the MCP server.

        Returns:
            Tuple containing the resource content as bytes and the MIME type

        """
        async with self._run_session() as session:
            return await session.read_resource(resource_name)

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
                if isinstance(message.content, types.TextContent):
                    llama_messages.append(
                        ChatMessage(
                            role=message.role,
                            blocks=[TextBlock(text=message.content.text)],
                        )
                    )
                elif isinstance(message.content, types.ImageContent):
                    llama_messages.append(
                        ChatMessage(
                            role=message.role,
                            blocks=[
                                ImageBlock(
                                    image=message.content.data,
                                    image_mimetype=message.content.mimeType,
                                )
                            ],
                        )
                    )
                elif isinstance(message.content, types.EmbeddedResource):
                    raise NotImplementedError(
                        "Embedded resources are not supported yet"
                    )
                else:
                    raise ValueError(
                        f"Unsupported content type: {type(message.content)}"
                    )

            return llama_messages
