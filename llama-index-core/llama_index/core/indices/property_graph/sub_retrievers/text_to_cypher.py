from typing import Any, List, Optional, Union

from llama_index.core.graph_stores.types import PropertyGraphStore
from llama_index.core.indices.property_graph.sub_retrievers.base import BasePGRetriever
from llama_index.core.llms import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.settings import Settings

DEFAULT_RESPONSE_TEMPLATE = (
    "Generated Cypher query:\n{query}\n\n" "Cypher Response:\n{response}"
)


class TextToCypherRetriever(BasePGRetriever):
    """A Text-to-Cypher retriever that uses a language model to generate Cypher queries.

    NOTE: Executing arbitrary cypher has its risks. Ensure you take the needed measures
    (read-only roles, sandboxed env, etc.) to ensure safe usage in a production environment.

    Args:
        graph_store (PropertyGraphStore):
            The graph store to retrieve data from.
        llm (Optional[LLM], optional):
            The language model to use. Defaults to Settings.llm.
        text_to_cypher_template (Optional[Union[PromptTemplate, str]], optional):
            The template to use for the text-to-cypher query. Defaults to None.
        response_template (Optional[str], optional):
            The template to use for the response. Defaults to None.
        cypher_validator (Optional[callable], optional):
            A callable function to validate the generated Cypher query. Defaults to None.
        allowed_query_fields (Optional[List[str]], optional):
            The fields to allow in the query output. Defaults to ["text", "label", "type"].
    """

    def __init__(
        self,
        graph_store: PropertyGraphStore,
        llm: Optional[LLM] = None,
        text_to_cypher_template: Optional[Union[PromptTemplate, str]] = None,
        response_template: Optional[str] = None,
        cypher_validator: Optional[callable] = None,
        allowed_output_fields: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        if not graph_store.supports_structured_queries:
            raise ValueError(
                "The provided graph store does not support cypher queries."
            )

        self.llm = llm or Settings.llm

        if isinstance(text_to_cypher_template, str):
            text_to_cypher_template = PromptTemplate(text_to_cypher_template)

        self.response_template = response_template or DEFAULT_RESPONSE_TEMPLATE
        self.text_to_cypher_template = (
            text_to_cypher_template or graph_store.text_to_cypher_template
        )
        self.cypher_validator = cypher_validator
        self.allowed_output_fields = allowed_output_fields
        super().__init__(graph_store=graph_store, include_text=False)

    def _parse_generated_cypher(self, cypher_query: str) -> str:
        if self.cypher_validator is not None:
            return self.cypher_validator(cypher_query)
        return cypher_query

    def _clean_query_output(self, query_output: Any) -> Any:
        """Iterate the cypher response, looking for the allowed fields."""
        if isinstance(query_output, dict):
            filtered = {}
            for key, value in query_output.items():
                if (
                    self.allowed_output_fields is None
                    or key in self.allowed_output_fields
                ):
                    filtered[key] = value
                elif isinstance(value, (dict, list)):
                    filtered_value = self._clean_query_output(value)
                    if filtered_value:
                        filtered[key] = filtered_value
            return filtered
        elif isinstance(query_output, list):
            filtered = []
            for item in query_output:
                filtered_item = self._clean_query_output(item)
                if filtered_item:
                    filtered.append(filtered_item)
            return filtered

        return None

    def retrieve_from_graph(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        schema = self._graph_store.get_schema_str()
        question = query_bundle.query_str

        response = self.llm.predict(
            self.text_to_cypher_template,
            schema=schema,
            question=question,
        )

        parsed_cypher_query = self._parse_generated_cypher(response)

        query_output = self._graph_store.structured_query(parsed_cypher_query)

        cleaned_query_output = self._clean_query_output(query_output)

        node_text = self.response_template.format(
            query=parsed_cypher_query,
            response=str(cleaned_query_output),
        )

        return [
            NodeWithScore(
                node=TextNode(
                    text=node_text,
                ),
                score=1.0,
            )
        ]

    async def aretrieve_from_graph(
        self, query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        schema = await self._graph_store.aget_schema_str()
        question = query_bundle.query_str

        response = await self.llm.apredict(
            self.text_to_cypher_template,
            schema=schema,
            question=question,
        )

        parsed_cypher_query = self._parse_generated_cypher(response)

        query_output = await self._graph_store.astructured_query(parsed_cypher_query)

        cleaned_query_output = self._clean_query_output(query_output)

        node_text = self.response_template.format(
            query=parsed_cypher_query,
            response=str(cleaned_query_output),
        )

        return [
            NodeWithScore(
                node=TextNode(
                    text=node_text,
                ),
                score=1.0,
            )
        ]
