"""Moss tool spec."""

from typing import List, Optional
from dataclasses import dataclass
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from inferedge_moss import MossClient, DocumentInfo


@dataclass
class QueryOptions:
    """
    Configuration options for Moss search queries.

    Attributes:
        top_k (int): Number of results to return from queries. Defaults to 5.
        alpha (float): Hybrid search weight (0.0=keyword, 1.0=semantic). Defaults to 0.5.
        model_id (str): The model ID to use for embeddings. Defaults to "moss-minilm".

    """

    top_k: int = 5
    alpha: float = 0.5
    model_id: str = "moss-minilm"


class MossToolSpec(BaseToolSpec):
    """
    Moss Tool Spec.

    This tool allows agents to interact with the Moss search engine to index documents
    and query for relevant information.
    """

    spec_functions: tuple[str, ...] = ("query",)

    def __init__(
        self,
        client: MossClient,
        index_name: str,
        query_options: Optional[QueryOptions] = None,
    ) -> None:
        """
        Initialize the Moss tool spec.

        Args:
            client (MossClient): The client to interact with the Moss service.
            index_name (str): The name of the index to use.
            query_options (Optional[QueryOptions]): Configuration options for the tool.
                Includes top_k (int), alpha (float), and model_id (str).

        """
        opt = query_options or QueryOptions()

        if not (0.0 <= opt.alpha <= 1.0):
            raise ValueError("alpha must be between 0 and 1")
        if opt.top_k < 1:
            raise ValueError("top_k must be greater than 0")

        self.top_k: int = opt.top_k
        self.alpha: float = opt.alpha
        self.client: MossClient = client
        self.index_name: str = index_name
        self._index_loaded: bool = False
        self.model_id: str = opt.model_id

    async def index_docs(self, docs: List[DocumentInfo]) -> None:
        await self.client.create_index(self.index_name, docs, model_id=self.model_id)
        self._index_loaded = False

    async def _load_index(self) -> None:
        """Load the index if it hasn't been loaded locally yet."""
        await self.client.load_index(self.index_name)
        self._index_loaded = True

    async def query(self, query: str) -> str:
        """
        Search the Moss knowledge base for information relevant to a specific query.

        This tool performs a hybrid semantic search to find the most relevant
        text snippets from the indexed documents. It is best used for answering
        technical questions, retrieving facts, or finding specific context
        within a large collection of documents.

        Args:
            query (str): The search terms or question to look up in the index.

        Returns:
            str: A formatted report containing the top matching text snippets,
                 their relevance scores, and their source metadata (like filename).

        """
        if not self._index_loaded:
            await self._load_index()

        results = await self.client.query(
            self.index_name, query, self.top_k, self.alpha
        )
        answer = f"Search results for: '{query}'\n\n"

        for i, result in enumerate(results.docs):
            source = (
                result.metadata.get("filename")
                or result.metadata.get("source")
                or "Unknown Source"
            )
            page = result.metadata.get("page", "N/A")

            answer += f"Match {i + 1} [Score: {result.score:.2f}]\n"
            answer += f"Source: {source} (Page: {page})\n"
            answer += f"Content: {result.text}\n"
            answer += "-" * 20 + "\n\n"

        return answer
