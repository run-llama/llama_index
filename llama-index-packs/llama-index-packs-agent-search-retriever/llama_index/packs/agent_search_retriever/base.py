from enum import Enum
from typing import Any, Dict, List, Optional

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.core.schema import NodeWithScore, TextNode


class SearchProvider(str, Enum):
    BING = "bing"
    AGENT_SEARCH = "agent-search"


class AgentSearchRetriever(BaseRetriever):
    """Retriever that uses the Agent Search API to retrieve documents."""

    def __init__(
        self,
        search_provider: str = "agent-search",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        similarity_top_k: int = 4,
    ) -> None:
        import_err_msg = (
            "`agent-search` package not found, please run `pip install agent-search`"
        )
        try:
            import agent_search  # noqa: F401
        except ImportError:
            raise ImportError(import_err_msg)

        from agent_search import SciPhi

        self._client = SciPhi(api_base=api_base, api_key=api_key)
        self._search_provider = SearchProvider(search_provider)
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        search_result = self._client.search(
            query_bundle.query_str, search_provider=self._search_provider.value
        )
        nodes = []
        found_texts = set()
        for result in search_result:
            if result["text"] in found_texts:
                continue
            found_texts.add(result["text"])

            metadata = {}
            metadata["url"] = result["url"]
            metadata["title"] = result["title"]
            nodes.append(
                NodeWithScore(
                    node=TextNode(
                        text=result["text"],
                        score=result["score"],
                        metadata=result["metadata"],
                    ),
                    score=result["score"],
                )
            )

        return nodes[: self._similarity_top_k]


class AgentSearchRetrieverPack(BaseLlamaPack):
    """AgentSearchRetrieverPack for running an agent-search retriever."""

    def __init__(
        self,
        similarity_top_k: int = 2,
        search_provider: str = "agent-search",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        self.retriever = AgentSearchRetriever(
            search_provider=search_provider,
            api_key=api_key,
            api_base=api_base,
            similarity_top_k=similarity_top_k,
        )
        super().__init__()

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "retriever": self.retriever,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return self._retriever.retrieve(*args, **kwargs)
