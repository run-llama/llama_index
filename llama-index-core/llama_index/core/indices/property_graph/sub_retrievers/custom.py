from abc import abstractmethod
from typing import Any, List

from llama_index.core.graph_stores.types import PropertyGraphStore
from llama_index.core.indices.property_graph.sub_retrievers.base import BasePGRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode


class CustomPGRetriever(BasePGRetriever):
    """A retriever meant to be easily subclassed to implement custom retrieval logic.

    The user only has to implement:
    - `init` to initialize the retriever and assign any necessary attributes.
    - `custom_retrieve` to implement the custom retrieval logic.
    - `aretrieve_custom` (optional) to implement asynchronous retrieval logic.

    Args:
        graph_store (PropertyGraphStore):
            The graph store to retrieve data from.
    """

    def __init__(
        self,
        graph_store: PropertyGraphStore,
        **kwargs: Any,
    ) -> None:
        self.init(**kwargs)
        super().__init__(graph_store=graph_store, include_text=False, **kwargs)

    @abstractmethod
    def init(self, **kwargs: Any):
        ...

    @abstractmethod
    def custom_retrieve(self, query_str: str) -> str:
        ...

    async def acustom_retrieve(self, query_str: str) -> str:
        return self.retrieve_custom(query_str)

    def retrieve_from_graph(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        result_str = self.retrieve_custom(query_bundle.query_str)
        return [NodeWithScore(node=TextNode(text=result_str), score=1.0)]

    async def aretrieve_from_graph(
        self, query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        result_str = await self.aretrieve_custom(query_bundle.query_str)
        return [NodeWithScore(node=TextNode(text=result_str), score=1.0)]
