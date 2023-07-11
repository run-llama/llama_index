""" Knowledge Graph Query Engine"""

import logging
from typing import Any, Optional

from llama_index.bridge.langchain import print_text
from llama_index.graph_stores.registery import (
    GRAPH_STORE_CLASS_TO_GRAPH_STORE_TYPE,
    GraphStoreType,
)
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts.base import Prompt, PromptType
from llama_index.response.schema import RESPONSE_TYPE, Response
from llama_index.storage.storage_context import StorageContext

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
DEFAULT_NEBULAGRAPH_NL2CYPHER_PROMPT = Prompt(
    DEFAULT_NEBULAGRAPH_NL2CYPHER_PROMPT_TMPL,
    prompt_type=PromptType.TEXT_TO_GRAPH_QUERY,
)

DEFAULT_NL2GRAPH_PROMPT_MAP = {
    GraphStoreType.NEBULA: DEFAULT_NEBULAGRAPH_NL2CYPHER_PROMPT,
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

DEFAULT_KG_RESPONSE_ANSWER_PROMPT = Prompt(
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
        **kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        service_context: Optional[ServiceContext] = None,
        storage_context: Optional[StorageContext] = None,
        graph_query_synthesis_prompt: Optional[Prompt] = None,
        graph_response_answer_prompt: Optional[Prompt] = None,
        refresh_schema: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ):
        # Ensure that we have a graph store
        assert storage_context is not None, "Must provide a storage context."
        assert (
            storage_context.graph_store is not None
        ), "Must provide a graph store in the storage context."
        self.storage_context = storage_context
        self.graph_store = storage_context.graph_store

        self.service_context = service_context or ServiceContext.from_defaults()

        # Get Graph Store Type
        self.graph_store_type = GRAPH_STORE_CLASS_TO_GRAPH_STORE_TYPE[
            self.graph_store.__class__
        ]

        # Get Graph schema
        self.graph_schema = self.graph_store.get_schema(refresh=refresh_schema)

        # Get graph store query synthesis prompt
        self.graph_query_synthesis_prompt = (
            graph_query_synthesis_prompt
            or DEFAULT_NL2GRAPH_PROMPT_MAP[self.graph_store_type]
        )

        self.graph_response_answer_prompt = (
            graph_response_answer_prompt or DEFAULT_KG_RESPONSE_ANSWER_PROMPT
        )
        self.verbose = verbose

        super().__init__(self.service_context.callback_manager)

    def generate_query(self, query_str: str) -> str:
        """Generate a Graph Store Query from a query bundle."""
        # Get the query engine query string

        graph_store_query: str = self.service_context.llm_predictor.predict(
            self.graph_query_synthesis_prompt,
            query_str=query_str,
            schema=self.graph_schema,
        )

        return graph_store_query

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Query the graph store."""
        graph_store_query = self.generate_query(query_bundle.query_str)
        if self.verbose:
            print_text(f"Graph Store Query: {graph_store_query}\n", color="yellow")
        logger.info(f"Graph Store Query: {graph_store_query}")
        # Get the graph store response
        graph_store_response = self.graph_store.query(query=graph_store_query)
        if self.verbose:
            print_text(
                f"Graph Store Response: {graph_store_response}\n", color="yellow"
            )
        logger.info(f"Graph Store Response: {graph_store_response}")
        response_str = self.service_context.llm_predictor.predict(
            self.graph_response_answer_prompt,
            query_str=query_bundle.query_str,
            kg_query_str=graph_store_query,
            kg_response_str=graph_store_response,
        )
        if self.verbose:
            print_text(f"Final Response: {response_str}\n", color="green")
        response_metadata = {
            "query_str": query_bundle.query_str,
            "graph_store_query": graph_store_query,
            "graph_store_response": graph_store_response,
            "graph_schema": self.graph_schema,
        }

        return Response(
            response_str,
            metadata=response_metadata,
        )

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Not supported."""
        return self._query(query_bundle)
