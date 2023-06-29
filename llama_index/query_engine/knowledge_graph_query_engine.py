""" Knowledge Graph Query Engine"""

from typing import Optional

from llama_index.bridge.langchain import (
    BaseLanguageModel,
    GraphCypherQAChain,
    KuzuGraph,
    KuzuQAChain,
    NebulaGraph,
    NebulaGraphQAChain,
    Neo4jGraph,
)
from llama_index.callbacks.base import CallbackManager
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.service_context import ServiceContext
from llama_index.langchain_helpers.chain_wrapper import LLMPredictor
from llama_index.response.schema import RESPONSE_TYPE

GRAPH_MAPPING = {
    "neo4jgraph": {
        "graph": Neo4jGraph,
        "qa_chain": GraphCypherQAChain,
    },
    "nebulagraph": {
        "graph": NebulaGraph,
        "qa_chain": NebulaGraphQAChain,
    },
    "kuzugraph": {
        "graph": KuzuGraph,
        "qa_chain": KuzuQAChain,
    },
}


class KnowledgeGraphQueryEngine(BaseQueryEngine):
    """Knowledge graph query engine.

    Query engine too call a knowledge graph.

    Args:
        graph (str): The name of the graph to query, a key in GRAPH_MAPPING.
        graph_kwargs (Optional[dict]): Keyword arguments to pass to the graph.
        qa_chain_kwargs (Optional[dict]): Keyword arguments to pass to the QA chain.
        llm (Optional[BaseLanguageModel]): A LLM to use for QA.
        llm_predictor (Optional[LLMPredictor]): A LLM predictor to use for QA.
        service_context (Optional[ServiceContext]): A service context to use.

    """

    def __init__(
        self,
        graph: str,
        graph_kwargs: Optional[dict] = None,
        qa_chain_kwargs: Optional[dict] = None,
        llm: Optional[BaseLanguageModel] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        service_context: Optional[ServiceContext] = None,
    ):
        # assert graph is not None and one of the graphs in GRAPH_MAPPING
        assert (
            graph is not None and graph in GRAPH_MAPPING.keys()
        ), f"Must provide a graph in {GRAPH_MAPPING.keys()}."
        # assert llm, llm_predictor, or service_context is not None
        assert any(
            [llm, llm_predictor, service_context]
        ), "Must provide either a LLM, LLM predictor, or service context."
        # assert only one of llm, llm_predictor, or service_context is not None
        assert (
            sum([bool(llm), bool(llm_predictor), bool(service_context)]) == 1
        ), "Must provide only one of LLM, LLM predictor, or service context."

        self._llm = llm
        self._llm_predictor = llm_predictor
        self.service_context = service_context or ServiceContext.from_defaults()

        # the priority to get LLM is: llm > llm_predictor > service_context
        if self._llm is None:
            if self._llm_predictor is None:
                self._llm = self.service_context.llm_predictor.llm  # type: ignore
            else:
                self._llm = self._llm_predictor.llm

        # graph and qa_chain
        self._graph = GRAPH_MAPPING[graph]["graph"](**(graph_kwargs or {}))

        qa_chain = GRAPH_MAPPING[graph]["qa_chain"]
        self._qa_chain = qa_chain.from_llm(
            llm=self._llm, graph=self._graph, **(qa_chain_kwargs or {})
        )

        super().__init__(self.service_context.callback_manager)

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        query_str = query_bundle.query_str
        return self._qa_chain.run(query_str)

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Not supported."""
        return self._query(query_bundle)
