import os
from typing import List, Optional
from uuid import UUID
from gestell import Gestell
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core.tools import FunctionTool


class GestellToolSpec(BaseToolSpec):
    """
    A tool spec that lets llamaindex agents work with Gestell's
    native `search` and streaming `prompt` endpoints.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        collection_id: Optional[str] = None,
    ) -> None:
        self._client: Gestell = Gestell(key=api_key or os.getenv("GESTELL_API_KEY", ""))
        self.collection_id: str = collection_id or os.getenv(
            "GESTELL_COLLECTION_ID", ""
        )
        super().__init__()

    @property
    def client(self) -> Gestell:
        """
        Access to the Gestell SDK client.
        Review https://gestell.ai/docs/reference to learn more.
        """
        return self._client

    def validate_collection_id(self, collection_id: Optional[str]) -> str:
        """
        Returns collection_id if it's a valid UUID, otherwise falls back to self.collection_id.
        """
        try:
            if collection_id:
                UUID(collection_id)
                return collection_id
        except ValueError:
            pass
        return self.collection_id

    async def aprompt(
        self,
        prompt: str,
        collection_id: Optional[str] = None,
        category_id: Optional[str] = None,
        method: Optional[str] = None,
        search_type: Optional[str] = None,
        vector_depth: Optional[int] = None,
        node_depth: Optional[int] = None,
        max_queries: Optional[int] = None,
        max_results: Optional[int] = None,
        template: Optional[str] = None,
        cot: Optional[bool] = None,
        messages: Optional[List[dict]] = None,
    ) -> str:
        """
        Execute a prompt, review https://gestell.ai/docs/reference#query for usage.

        :param prompt: The prompt or query text to execute.
        :param collection_id: The UUID of the collection to query (or set via GESTELL_COLLECTION_ID).
        :param category_id: An optional category UUID to filter results by.
        :param method: The search method to use: choose between 'fast', 'normal' and 'precise'
        :param search_type: The search type to specify: choose between 'keywords', 'phrase' and 'summary'
        :param vector_depth: Depth of vector-based retrieval.
        :param node_depth: Depth of node-based retrieval.
        :param max_queries: Maximum number of sub-queries to run.
        :param max_results: Maximum number of results to return.
        :param template: A prompt template to apply before sending.
        :param cot: Whether to enable chain-of-thought reasoning.
        :param messages: The message history for streaming chat contexts.
        :returns: The full response text.
        """
        stream = self._client.query.prompt(
            collection_id=self.validate_collection_id(collection_id),
            prompt=prompt,
            category_id=category_id,
            method=method,
            type=search_type,
            vectorDepth=vector_depth,
            nodeDepth=node_depth,
            maxQueries=max_queries,
            maxResults=max_results,
            template=template,
            cot=cot,
            messages=messages,
        )
        tokens: List[str] = []
        async for token in stream:
            tokens.append(token.decode())

        return "".join(tokens)

    async def asearch(
        self,
        query: str,
        collection_id: Optional[str] = None,
        category_id: Optional[str] = None,
        method: Optional[str] = None,
        search_type: Optional[str] = None,
        vector_depth: Optional[int] = None,
        node_depth: Optional[int] = None,
        max_queries: Optional[int] = None,
        max_results: Optional[int] = None,
        include_content: Optional[bool] = None,
        include_edges: Optional[bool] = None,
    ) -> List[dict]:
        """
        Execute a search query, review https://gestell.ai/docs/reference#query for usage.

        :param query: The search query or prompt text.
        :param collection_id: The UUID of the collection to query (or set via GESTELL_COLLECTION_ID).
        :param category_id: An optional category UUID to filter results by.
        :param method: The search method to use: choose between 'fast', 'normal' and 'precise'
        :param search_type: The search type to specify: choose between 'keywords', 'phrase' and 'summary'
        :param vector_depth: Depth of vector-based retrieval.
        :param node_depth: Depth of node-based retrieval.
        :param max_queries: Maximum number of sub-queries to run.
        :param max_results: Maximum number of results to return.
        :param include_content: Whether to include the full content in each result.
        :param include_edges: Whether to include edge metadata in results.
        :returns: A list of search result dictionaries.
        """
        resp = await self._client.query.search(
            collection_id=self.validate_collection_id(collection_id),
            prompt=query,
            category_id=category_id,
            method=method,
            type=search_type,
            vectorDepth=vector_depth,
            nodeDepth=node_depth,
            maxQueries=max_queries,
            maxResults=max_results,
            includeContent=include_content,
            includeEdges=include_edges,
        )
        return resp.result

    def to_tool_list(self):
        """
        Returns the FunctionTool definitions for Gestell's search and prompt APIs.

        Returns:
        list[FunctionTool]:
        - 'gestell_search': to run a search query and return structured JSON results.
        - 'gestell_prompt': to stream prompt responses token by token directly to the agent.

        """
        return [
            FunctionTool.from_defaults(
                self.asearch,
                name="gestell_search",
                description="Search Gestell collection content and return JSON results",
            ),
            FunctionTool.from_defaults(
                self.aprompt,
                name="gestell_prompt",
                description="Stream a prompt response from a Gestell collection",
            ),
        ]
