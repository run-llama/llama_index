""" Knowledge Graph Query Engine."""

import logging
from typing import Any, Dict, List, Optional, Sequence

from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.core import BaseQueryEngine
from llama_index.graph_stores.registry import (
    GRAPH_STORE_CLASS_TO_GRAPH_STORE_TYPE,
    GraphStoreType,
)
from llama_index.prompts.base import BasePromptTemplate, PromptTemplate, PromptType
from llama_index.prompts.mixin import PromptDictType, PromptMixinType
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.response_synthesizers import BaseSynthesizer, get_response_synthesizer
from llama_index.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.service_context import ServiceContext
from llama_index.storage.storage_context import StorageContext
from llama_index.utils import print_text

logger = logging.getLogger(__name__)

# Prompt
DEFAULT_NEBULAGRAPH_NL2CYPHER_PROMPT_TMPL = """
Generate NebulaGraph query from natural language.
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
---
{schema}
---
Note: NebulaGraph speaks a dialect of Cypher, comparing to standard Cypher:

1. it uses double equals sign for comparison: `==` rather than `=`
2. it needs explicit label specification when referring to node properties, i.e.
v is a variable of a node, and we know its label is Foo, v.`foo`.name is correct
while v.name is not.

For example, see this diff between standard and NebulaGraph Cypher dialect:
```diff
< MATCH (p:person)-[:directed]->(m:movie) WHERE m.name = 'The Godfather'
< RETURN p.name;
---
> MATCH (p:`person`)-[:directed]->(m:`movie`) WHERE m.`movie`.`name` == 'The Godfather'
> RETURN p.`person`.`name`;
```

Question: {query_str}

NebulaGraph Cypher dialect query:
"""
DEFAULT_NEBULAGRAPH_NL2CYPHER_PROMPT = PromptTemplate(
    DEFAULT_NEBULAGRAPH_NL2CYPHER_PROMPT_TMPL,
    prompt_type=PromptType.TEXT_TO_GRAPH_QUERY,
)

# Prompt
DEFAULT_NEO4J_NL2CYPHER_PROMPT_TMPL = (
    "Task:Generate Cypher statement to query a graph database.\n"
    "Instructions:\n"
    "Use only the provided relationship types and properties in the schema.\n"
    "Do not use any other relationship types or properties that are not provided.\n"
    "Schema:\n"
    "{schema}\n"
    "Note: Do not include any explanations or apologies in your responses.\n"
    "Do not respond to any questions that might ask anything else than for you "
    "to construct a Cypher statement. \n"
    "Do not include any text except the generated Cypher statement.\n"
    "\n"
    "The question is:\n"
    "{query_str}\n"
)

DEFAULT_NEO4J_NL2CYPHER_PROMPT = PromptTemplate(
    DEFAULT_NEO4J_NL2CYPHER_PROMPT_TMPL,
    prompt_type=PromptType.TEXT_TO_GRAPH_QUERY,
)

DEFAULT_NL2GRAPH_PROMPT_MAP = {
    GraphStoreType.NEBULA: DEFAULT_NEBULAGRAPH_NL2CYPHER_PROMPT,
    GraphStoreType.NEO4J: DEFAULT_NEO4J_NL2CYPHER_PROMPT,
}

DEFAULT_KG_RESPONSE_ANSWER_PROMPT_TMPL = """
The original question is given below.
This question has been translated into a Graph Database query.
Both the Graph query and the response are given below.
Given the Graph Query response, synthesise a response to the original question.

Original question: {query_str}
Graph query: {kg_query_str}
Graph response: {kg_response_str}
Response:
"""

DEFAULT_KG_RESPONSE_ANSWER_PROMPT = PromptTemplate(
    DEFAULT_KG_RESPONSE_ANSWER_PROMPT_TMPL,
    prompt_type=PromptType.QUESTION_ANSWER,
)


class KnowledgeGraphQueryEngine(BaseQueryEngine):
    """Knowledge graph query engine.

    Query engine to call a knowledge graph.

    Args:
        service_context (Optional[ServiceContext]): A service context to use.
        storage_context (Optional[StorageContext]): A storage context to use.
        refresh_schema (bool): Whether to refresh the schema.
        verbose (bool): Whether to print intermediate results.
        response_synthesizer (Optional[BaseSynthesizer]):
            A BaseSynthesizer object.
        **kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        service_context: Optional[ServiceContext] = None,
        storage_context: Optional[StorageContext] = None,
        graph_query_synthesis_prompt: Optional[BasePromptTemplate] = None,
        graph_response_answer_prompt: Optional[BasePromptTemplate] = None,
        refresh_schema: bool = False,
        verbose: bool = False,
        response_synthesizer: Optional[BaseSynthesizer] = None,
        **kwargs: Any,
    ):
        # Ensure that we have a graph store
        assert storage_context is not None, "Must provide a storage context."
        assert (
            storage_context.graph_store is not None
        ), "Must provide a graph store in the storage context."
        self._storage_context = storage_context
        self.graph_store = storage_context.graph_store

        self._service_context = service_context or ServiceContext.from_defaults()

        # Get Graph Store Type
        self._graph_store_type = GRAPH_STORE_CLASS_TO_GRAPH_STORE_TYPE[
            self.graph_store.__class__
        ]

        # Get Graph schema
        self._graph_schema = self.graph_store.get_schema(refresh=refresh_schema)

        # Get graph store query synthesis prompt
        self._graph_query_synthesis_prompt = (
            graph_query_synthesis_prompt
            or DEFAULT_NL2GRAPH_PROMPT_MAP[self._graph_store_type]
        )

        self._graph_response_answer_prompt = (
            graph_response_answer_prompt or DEFAULT_KG_RESPONSE_ANSWER_PROMPT
        )
        self._verbose = verbose
        self._response_synthesizer = response_synthesizer or get_response_synthesizer(
            callback_manager=self._service_context.callback_manager,
            service_context=self._service_context,
        )

        super().__init__(self._service_context.callback_manager)

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {
            "graph_query_synthesis_prompt": self._graph_query_synthesis_prompt,
            "graph_response_answer_prompt": self._graph_response_answer_prompt,
        }

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "graph_query_synthesis_prompt" in prompts:
            self._graph_query_synthesis_prompt = prompts["graph_query_synthesis_prompt"]
        if "graph_response_answer_prompt" in prompts:
            self._graph_response_answer_prompt = prompts["graph_response_answer_prompt"]

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {"response_synthesizer": self._response_synthesizer}

    def generate_query(self, query_str: str) -> str:
        """Generate a Graph Store Query from a query bundle."""
        # Get the query engine query string

        graph_store_query: str = self._service_context.llm_predictor.predict(
            self._graph_query_synthesis_prompt,
            query_str=query_str,
            schema=self._graph_schema,
        )

        return graph_store_query

    async def agenerate_query(self, query_str: str) -> str:
        """Generate a Graph Store Query from a query bundle."""
        # Get the query engine query string

        graph_store_query: str = await self._service_context.llm_predictor.apredict(
            self._graph_query_synthesis_prompt,
            query_str=query_str,
            schema=self._graph_schema,
        )

        return graph_store_query

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Get nodes for response."""
        graph_store_query = self.generate_query(query_bundle.query_str)
        if self._verbose:
            print_text(f"Graph Store Query:\n{graph_store_query}\n", color="yellow")
        logger.debug(f"Graph Store Query:\n{graph_store_query}")

        with self.callback_manager.event(
            CBEventType.RETRIEVE,
            payload={EventPayload.QUERY_STR: graph_store_query},
        ) as retrieve_event:
            # Get the graph store response
            graph_store_response = self.graph_store.query(query=graph_store_query)
            if self._verbose:
                print_text(
                    f"Graph Store Response:\n{graph_store_response}\n",
                    color="yellow",
                )
            logger.debug(f"Graph Store Response:\n{graph_store_response}")

            retrieve_event.on_end(payload={EventPayload.RESPONSE: graph_store_response})

        retrieved_graph_context: Sequence = self._graph_response_answer_prompt.format(
            query_str=query_bundle.query_str,
            kg_query_str=graph_store_query,
            kg_response_str=graph_store_response,
        )

        node = NodeWithScore(
            node=TextNode(
                text=retrieved_graph_context,
                score=1.0,
                metadata={
                    "query_str": query_bundle.query_str,
                    "graph_store_query": graph_store_query,
                    "graph_store_response": graph_store_response,
                    "graph_schema": self._graph_schema,
                },
            )
        )
        return [node]

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Query the graph store."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            nodes: List[NodeWithScore] = self._retrieve(query_bundle)

            response = self._response_synthesizer.synthesize(
                query=query_bundle,
                nodes=nodes,
            )

            if self._verbose:
                print_text(f"Final Response: {response}\n", color="green")

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        graph_store_query = await self.agenerate_query(query_bundle.query_str)
        if self._verbose:
            print_text(f"Graph Store Query:\n{graph_store_query}\n", color="yellow")
        logger.debug(f"Graph Store Query:\n{graph_store_query}")

        with self.callback_manager.event(
            CBEventType.RETRIEVE,
            payload={EventPayload.QUERY_STR: graph_store_query},
        ) as retrieve_event:
            # Get the graph store response
            # TBD: This is a blocking call. We need to make it async.
            graph_store_response = self.graph_store.query(query=graph_store_query)
            if self._verbose:
                print_text(
                    f"Graph Store Response:\n{graph_store_response}\n",
                    color="yellow",
                )
            logger.debug(f"Graph Store Response:\n{graph_store_response}")

            retrieve_event.on_end(payload={EventPayload.RESPONSE: graph_store_response})

        retrieved_graph_context: Sequence = self._graph_response_answer_prompt.format(
            query_str=query_bundle.query_str,
            kg_query_str=graph_store_query,
            kg_response_str=graph_store_response,
        )

        node = NodeWithScore(
            node=TextNode(
                text=retrieved_graph_context,
                score=1.0,
                metadata={
                    "query_str": query_bundle.query_str,
                    "graph_store_query": graph_store_query,
                    "graph_store_response": graph_store_response,
                    "graph_schema": self._graph_schema,
                },
            )
        )
        return [node]

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Query the graph store."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            nodes = await self._aretrieve(query_bundle)
            response = await self._response_synthesizer.asynthesize(
                query=query_bundle,
                nodes=nodes,
            )

            if self._verbose:
                print_text(f"Final Response: {response}\n", color="green")

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response
