from typing import List, Optional, Union

from llama_index.core.llms.llm import LLM
from llama_index.core.indices.property_graph.base import (
    TRIPLET_SOURCE_KEY,
)
from llama_index.core.indices.property_graph.sub_retrievers.base import (
    BaseLPGRetriever,
)
from llama_index.core.graph_stores.types import LabelledPropertyGraphStore, Triplet
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.core.settings import Settings
from llama_index.core.schema import (
    NodeWithScore,
    TextNode,
    NodeRelationship,
    RelatedNodeInfo,
    QueryBundle,
)

DEFAULT_SYNONYM_EXPAND_TEMPLATE = (
    "Given some initial query, generate synonyms or related keywords up to {max_keywords} in total, "
    "considering possible cases of capitalization, pluralization, common expressions, etc.\n"
    "Provide all synonyms/keywords separated by '^' symbols: 'keyword1^keyword2^...'\n"
    "Note, result should be in one-line, separated by '^' symbols."
    "----\n"
    "QUERY: {query_str}\n"
    "----\n"
    "KEYWORDS: "
)


class LLMSynonymRetriever(BaseLPGRetriever):
    def __init__(
        self,
        graph_store: LabelledPropertyGraphStore,
        include_text: bool = True,
        synonym_prompt: Union[
            BasePromptTemplate, str
        ] = DEFAULT_SYNONYM_EXPAND_TEMPLATE,
        max_keywords: int = 10,
        output_parsing_fn: Optional[callable] = None,
        llm: Optional[LLM] = None,
        **kwargs,
    ) -> None:
        self._llm = llm or Settings.llm
        if isinstance(synonym_prompt, str):
            synonym_prompt = PromptTemplate(synonym_prompt)
        self._synonym_prompt = synonym_prompt
        self._output_parsing_fn = output_parsing_fn
        self._max_keywords = max_keywords
        super().__init__(graph_store=graph_store, include_text=include_text, **kwargs)

    def _parse_llm_output(self, output: str) -> List[str]:
        if self._output_parsing_fn:
            matches = self._output_parsing_fn(output)
        else:
            matches = output.strip().split("^")

        # capitalize to normalize with ingestion
        return [x.strip().capitalize() for x in matches if x.strip()]

    def _get_nodes_with_score(self, triplets: List[Triplet]) -> List[NodeWithScore]:
        results = []
        for triplet in triplets:
            source_id = triplet[0].properties.get(TRIPLET_SOURCE_KEY, None)
            relationships = {}
            if source_id is not None:
                relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                    node_id=source_id
                )

            text = f"{triplet[0]!s} -> {triplet[1]!s} -> {triplet[2]!s}"
            results.append(
                NodeWithScore(
                    node=TextNode(
                        text=text,
                        relationships=relationships,
                    ),
                    score=1.0,
                )
            )

        return results

    def _prepare_matches(self, matches: List[Triplet]) -> List[NodeWithScore]:
        results = []
        for match in matches:
            sub_results = self._graph_store.get_triplets(entity_names=[match])
            results.extend(self._get_nodes_with_score(sub_results))

        return results

    async def _aprepare_matches(self, matches: List[str]) -> List[NodeWithScore]:
        results = []
        for match in matches:
            sub_results = await self._graph_store.aget_triplets(entity_names=[match])
            results.extend(self._get_nodes_with_score(sub_results))

        return results

    def retrieve_from_graph(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        response = self._llm.predict(
            self._synonym_prompt,
            query_str=query_bundle.query_str,
            max_keywords=self._max_keywords,
        )
        matches = self._parse_llm_output(response)

        return self._prepare_matches(matches)

    async def aretrieve_from_graph(
        self, query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        response = await self._llm.apredict(
            self._synonym_prompt,
            query_str=query_bundle.query_str,
            max_keywords=self._max_keywords,
        )
        matches = self._parse_llm_output(response)

        return await self._aprepare_matches(matches)
