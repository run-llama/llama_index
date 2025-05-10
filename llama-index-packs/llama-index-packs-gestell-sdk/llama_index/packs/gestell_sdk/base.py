import os
import asyncio
from typing import Optional, AsyncGenerator, List
from gestell import Gestell
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.tools import FunctionTool

GESTELL_API_KEY = "GESTELL_API_KEY"
GESTELL_COLLECTION_ID = "GESTELL_COLLECTION_ID"


class GestellSDKPack(BaseLlamaPack):
    """
    A tool calling llamapack that lets llamaindex agents work with Gestell's
    native `search` and streaming `prompt` endpoints.
    """

    def __init__(
        self,
        collection_id: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv(GESTELL_API_KEY)
        if not self.api_key:
            raise ValueError(f"Provide `api_key` or set {GESTELL_API_KEY}")

        self.collection_id = collection_id or os.getenv(GESTELL_COLLECTION_ID)
        if not self.collection_id:
            raise ValueError(f"Provide `collection_id` or set {GESTELL_COLLECTION_ID}")

        self._gestell = Gestell(key=self.api_key)  # one shared client

    async def aprompt(
        self,
        prompt: str,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Gestell SDK `prompt` endpoint parameters:
        collection_id (str)         - The ID of the collection to query (or set via GESTELL_COLLECTION_ID).
        prompt (str)                - The prompt or query text to execute.
        category_id (Optional[str]) - An optional category ID to filter results by.
        method (Optional[SearchMethod])
                                    - The search method to use (e.g. “fast”, “accurate”).
        type (Optional[SearchType]) - The search type to specify (e.g. “vector”, “node”).
        vectorDepth (Optional[int]) - Depth of vector-based retrieval.
        nodeDepth (Optional[int])   - Depth of node-based retrieval.
        maxQueries (Optional[int])  - Maximum number of sub-queries to run.
        maxResults (Optional[int])  - Maximum number of results to return.
        template (Optional[str])    - A prompt template to apply before sending.
        cot (Optional[bool])        - Whether to enable chain-of-thought reasoning.
        messages (Optional[List[PromptMessage]])
                                    - The message history for streaming chat contexts.
        """
        stream = self._gestell.query.prompt(self.collection_id, prompt=prompt, **kwargs)
        async for chunk in stream:
            yield chunk.decode()

    async def asearch(self, query: str, **kwargs) -> List[dict]:
        """
        Gestell SDK `search` endpoint parameters:
          collection_id (str)         - The ID of the collection to query (or set via GESTELL_COLLECTION_ID).
          prompt (str)                - The search query or prompt text.
          category_id (Optional[str]) - An optional category ID to filter results by.
          method (Optional[SearchMethod])
                                      - The search method to use.
          type (Optional[SearchType]) - The search type to specify.
          vectorDepth (Optional[int]) - Depth of vector-based retrieval.
          nodeDepth (Optional[int])   - Depth of node-based retrieval.
          maxQueries (Optional[int])  - Maximum number of sub-queries to run.
          maxResults (Optional[int])  - Maximum number of results to return.
          includeContent (Optional[bool])
                                      - Whether to include the full content in each result.
          includeEdges (Optional[bool])
                                      - Whether to include edge metadata in results.
        """
        resp = await self._gestell.query.search(
            self.collection_id, prompt=query, **kwargs
        )
        return resp.result

    def run(self, question: str) -> str:
        """
        Execute a single prompt query in a blocking, synchronous context.
        """
        return asyncio.run(self._sync_run(question))

    async def _sync_run(self, question: str) -> str:
        chunks = []
        async for part in self.aprompt(question):
            chunks.append(part)
        return "".join(chunks)

    def get_tools(self):
        """
        Returns the FunctionTool definitions for Gestell's search and prompt APIs.

        Returns:
        list[FunctionTool]:
        - 'gestell_search': to run a search query and return structured JSON results.
        - 'gestell_prompt': to stream prompt responses token by token directly to the agent.

        """
        return [
            FunctionTool.from_defaults(
                name="gestell_search",
                description="Search Gestell collection content and return JSON",
                fn=self.asearch,
            ),
            FunctionTool.from_defaults(
                name="gestell_prompt",
                description="Stream a prompt answer from Gestell collection",
                fn=self.aprompt,
                return_direct=True,  # agent returns text immediately
            ),
        ]
