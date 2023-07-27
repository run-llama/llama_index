from typing import Dict, List, Optional, Tuple, Union

from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.schema import QueryBundle
from llama_index.schema import TextNode, IndexNode, NodeWithScore
from llama_index.bridge.langchain import print_text
from llama_index.indices.base_retriever import BaseRetriever


DEFAULT_QUERY_RESPONSE_TMPL = "Query: {query_str}\nResponse: {response}"


R_AND_Q_TYPE = Union[BaseRetriever, BaseQueryEngine]


class RecursiveRetriever(BaseRetriever):
    """Recursive retriever.

    This retriever will recursively explore links from nodes to other
    retrievers/query engines.

    For any retrieved nodes, if any of the nodes are IndexNodes,
    then it will explore the linked retriever/query engine, and query that.

    Args:
        root_id (str): The root id of the query graph.
        retriever_dict (Optional[Dict[str, BaseRetriever]]): A dictionary
            of id to retrievers.
        query_engine_dict (Optional[Dict[str, BaseQueryEngine]]): A dictionary of
            id to query engines.

    """

    def __init__(
        self,
        root_id: str,
        retriever_dict: Dict[str, BaseRetriever],
        query_engine_dict: Optional[Dict[str, BaseQueryEngine]] = None,
        callback_manager: Optional[CallbackManager] = None,
        query_response_tmpl: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        """Init params."""
        self._root_id = root_id
        if root_id not in retriever_dict:
            raise ValueError(
                f"Root id {root_id} not in retriever_dict, it must be a retriever."
            )
        self._retriever_dict = retriever_dict
        self._query_engine_dict = query_engine_dict or {}
        self.callback_manager = callback_manager or CallbackManager([])

        # make sure keys don't overlap
        if set(self._retriever_dict.keys()) & set(self._query_engine_dict.keys()):
            raise ValueError("Retriever and query engine ids must not overlap.")

        self._query_response_tmpl = query_response_tmpl or DEFAULT_QUERY_RESPONSE_TMPL
        self._verbose = verbose
        super().__init__()

    def _query_retrieved_nodes(
        self, query_bundle: QueryBundle, node_with_score: NodeWithScore
    ) -> Tuple[List[NodeWithScore], List[NodeWithScore]]:
        """Query for retrieved nodes.

        If node is an IndexNode, then recursively query the retriever/query engine.
        If node is a TextNode, then simply return the node.

        """
        node = node_with_score.node

        if isinstance(node, IndexNode):
            if self._verbose:
                print_text(
                    "Retrieved node with id, entering: " f"{node.index_id}\n",
                    color="pink",
                )
            retrieved_nodes, additional_nodes = self._retrieve_rec(
                query_bundle, query_id=node.index_id
            )
        else:
            assert isinstance(node, TextNode)
            if self._verbose:
                print_text(
                    "Retrieving text node: " f"{node.get_content()}\n",
                    color="pink",
                )
            retrieved_nodes = [node_with_score]
            additional_nodes = []
        return retrieved_nodes, additional_nodes

    def _fetch_retriever_or_query_engine(self, query_id: str) -> R_AND_Q_TYPE:
        """Fetch retriever or query engine."""
        retriever = self._retriever_dict.get(query_id, None)
        if retriever is not None:
            return retriever
        query_engine = self._query_engine_dict.get(query_id, None)
        if query_engine is not None:
            return query_engine
        raise ValueError(
            f"Query id {query_id} not found in either `retriever_dict` "
            "or `query_engine_dict`."
        )

    def _retrieve_rec(
        self, query_bundle: QueryBundle, query_id: Optional[str] = None
    ) -> Tuple[List[NodeWithScore], List[NodeWithScore]]:
        """Query recursively."""
        if self._verbose:
            print_text(
                f"Retrieving with query id {query_id}: {query_bundle.query_str}\n",
                color="blue",
            )
        query_id = query_id or self._root_id
        obj = self._fetch_retriever_or_query_engine(query_id)
        if isinstance(obj, BaseRetriever):
            with self.callback_manager.event(CBEventType.RETRIEVE) as event:
                event.on_start()
                nodes = obj.retrieve(query_bundle)
                event.on_end(payload={EventPayload.NODES: nodes})

            nodes_to_add = []
            additional_nodes = []
            for node_with_score in nodes:
                cur_nodes_to_add, node_additional_sources = self._query_retrieved_nodes(
                    query_bundle, node_with_score
                )
                nodes_to_add.extend(cur_nodes_to_add)
                additional_nodes.extend(node_additional_sources)

        elif isinstance(obj, BaseQueryEngine):
            sub_resp = obj.query(query_bundle)
            if self._verbose:
                print_text(
                    f"Got response: {str(sub_resp)}\n",
                    color="green",
                )
            # format with both the query and the response
            node_text = self._query_response_tmpl.format(
                query_str=query_bundle.query_str, response=str(sub_resp)
            )
            node = TextNode(text=node_text)
            nodes_to_add = [NodeWithScore(node=node, score=1.0)]
            additional_nodes = sub_resp.source_nodes
        else:
            raise ValueError("Must be a retriever or query engine.")

        return nodes_to_add, additional_nodes

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        retrieved_nodes, _ = self._retrieve_rec(query_bundle, query_id=None)
        return retrieved_nodes

    def retrieve_all(
        self, query_bundle: QueryBundle
    ) -> Tuple[List[NodeWithScore], List[NodeWithScore]]:
        """Retrieve all nodes.

        Unlike default `retrieve` method, this also fetches additional sources.

        """
        return self._retrieve_rec(query_bundle, query_id=None)
