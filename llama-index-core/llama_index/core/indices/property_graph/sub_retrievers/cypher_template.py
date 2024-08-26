from typing import Any, List, Optional

from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.graph_stores.types import PropertyGraphStore
from llama_index.core.indices.property_graph.sub_retrievers.base import BasePGRetriever
from llama_index.core.llms import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.settings import Settings


class CypherTemplateRetriever(BasePGRetriever):
    """A Cypher retriever that fills in params for a cypher query using an LLM.

    Args:
        graph_store (PropertyGraphStore):
            The graph store to retrieve data from.
        output_cls (BaseModel):
            The output class to use for the LLM.
            Should contain the params needed for the cypher query.
        cypher_query (str):
            The cypher query to use, with templated params.
        llm (Optional[LLM], optional):
            The language model to use. Defaults to Settings.llm.
    """

    def __init__(
        self,
        graph_store: PropertyGraphStore,
        output_cls: BaseModel,
        cypher_query: str,
        llm: Optional[LLM] = None,
        **kwargs: Any,
    ) -> None:
        if not graph_store.supports_structured_queries:
            raise ValueError(
                "The provided graph store does not support cypher queries."
            )

        self.llm = llm or Settings.llm
        self.output_cls = output_cls
        self.cypher_query = cypher_query

        super().__init__(graph_store=graph_store, include_text=False)

    def retrieve_from_graph(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        question = query_bundle.query_str

        response = self.llm.structured_predict(
            self.output_cls, PromptTemplate(question)
        )

        cypher_response = self._graph_store.structured_query(
            self.cypher_query,
            param_map=response.model_dump(),
        )

        return [
            NodeWithScore(
                node=TextNode(
                    text=str(cypher_response),
                ),
                score=1.0,
            )
        ]

    async def aretrieve_from_graph(
        self, query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        question = query_bundle.query_str

        response = await self.llm.astructured_predict(
            self.output_cls, PromptTemplate(question)
        )

        cypher_response = await self._graph_store.astructured_query(
            self.cypher_query,
            param_map=response.model_dump(),
        )

        return [
            NodeWithScore(
                node=TextNode(
                    text=str(cypher_response),
                ),
                score=1.0,
            )
        ]
