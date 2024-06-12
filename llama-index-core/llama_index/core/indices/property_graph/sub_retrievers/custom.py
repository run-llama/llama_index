from abc import abstractmethod
from typing import Any, List, Union

from llama_index.core.graph_stores.types import PropertyGraphStore
from llama_index.core.indices.property_graph.sub_retrievers.base import BasePGRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

CUSTOM_RETRIEVE_TYPE = Union[
    str, List[str], TextNode, List[TextNode], NodeWithScore, List[NodeWithScore]
]


class CustomPGRetriever(BasePGRetriever):
    """A retriever meant to be easily subclassed to implement custom retrieval logic.

    The user only has to implement:
    - `init` to initialize the retriever and assign any necessary attributes.
    - `custom_retrieve` to implement the custom retrieval logic.
    - `aretrieve_custom` (optional) to implement asynchronous retrieval logic.

    Args:
        graph_store (PropertyGraphStore):
            The graph store to retrieve data from.
        include_text (bool):
            Whether to include text in the retrieved nodes. Only works for kg nodes
            inserted by LlamaIndex.
        **kwargs:
            Additional keyword arguments passed to init().
    """

    def __init__(
        self,
        graph_store: PropertyGraphStore,
        include_text: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(graph_store=graph_store, include_text=include_text, **kwargs)
        self.init(**kwargs)

    @property
    def graph_store(self) -> PropertyGraphStore:
        return self._graph_store

    @abstractmethod
    def init(self, **kwargs: Any):
        """Initialize the retriever.

        Has access to all keyword arguments passed to the retriever, as well as:
        - `self.graph_store`: The graph store to retrieve data from.
        - `self.include_text``: Whether to include text in the retrieved nodes.
        """
        ...

    @abstractmethod
    def custom_retrieve(self, query_str: str) -> CUSTOM_RETRIEVE_TYPE:
        """Retrieve data from the graph store based on the query string.

        Args:
            query_str (str): The query string to retrieve data for.

        Returns:
            The retrieved data. The return type can be one of:
            - str: A single string.
            - List[str]: A list of strings.
            - TextNode: A single TextNode.
            - List[TextNode]: A list of TextNodes.
            - NodeWithScore: A single NodeWithScore.
            - List[NodeWithScore]: A list of NodeWithScores.
        """
        ...

    async def acustom_retrieve(self, query_str: str) -> CUSTOM_RETRIEVE_TYPE:
        """Asynchronously retrieve data from the graph store based on the query string.

        Args:
            query_str (str): The query string to retrieve data for.

        Returns:
            The retrieved data. The return type can be one of:
            - str: A single string.
            - List[str]: A list of strings.
            - TextNode: A single TextNode.
            - List[TextNode]: A list of TextNodes.
            - NodeWithScore: A single NodeWithScore.
            - List[NodeWithScore]: A list of NodeWithScores.
        """
        return self.custom_retrieve(query_str)

    def _parse_custom_return_type(
        self, result: CUSTOM_RETRIEVE_TYPE
    ) -> List[NodeWithScore]:
        if isinstance(result, str):
            return [NodeWithScore(node=TextNode(text=result), score=1.0)]
        elif isinstance(result, list):
            if all(isinstance(item, str) for item in result):
                return [
                    NodeWithScore(node=TextNode(text=item), score=1.0)
                    for item in result
                ]
            elif all(isinstance(item, TextNode) for item in result):
                return [NodeWithScore(node=item, score=1.0) for item in result]
            elif all(isinstance(item, NodeWithScore) for item in result):
                return result
            else:
                raise ValueError(
                    "Invalid return type. All items in the list must be of the same type."
                )
        elif isinstance(result, TextNode):
            return [NodeWithScore(node=result, score=1.0)]
        elif isinstance(result, NodeWithScore):
            return [result]
        else:
            raise ValueError(f"Invalid return type: {type(result)}")

    def retrieve_from_graph(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        result = self.custom_retrieve(query_bundle.query_str)
        return self._parse_custom_return_type(result)

    async def aretrieve_from_graph(
        self, query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        result = await self.acustom_retrieve(query_bundle.query_str)
        return self._parse_custom_return_type(result)
