from typing import Any, Callable, List, Optional, Union

from llama_index.core.llms.llm import LLM
from llama_index.core.indices.property_graph.sub_retrievers.base import (
    BasePGRetriever,
)
from llama_index.core.graph_stores.types import (
    PropertyGraphStore,
    KG_SOURCE_REL,
)
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.core.settings import Settings
from llama_index.core.schema import (
    NodeWithScore,
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


class LLMSynonymRetriever(BasePGRetriever):
    """A retriever that uses a language model to expand a query with synonyms.
    The synonyms are then used to retrieve nodes from a property graph.

    Args:
        graph_store (PropertyGraphStore):
            The graph store to retrieve data from.
        include_text (bool, optional):
            Whether to include source text in the retrieved nodes. Defaults to True.
        synonym_prompt (Union[BasePromptTemplate, str], optional):
            The template to use for the synonym expansion query.
            Defaults to DEFAULT_SYNONYM_EXPAND_TEMPLATE.
        max_keywords (int, optional):
            The maximum number of synonyms to generate. Defaults to 10.
        path_depth (int, optional):
            The depth of the path to retrieve for each node. Defaults to 1 (i.e. a triple).
        output_parsing_fn (Optional[callable], optional):
            A callable function to parse the output of the language model. Defaults to None.
        llm (Optional[LLM], optional):
            The language model to use. Defaults to Settings.llm.
    """

    def __init__(
        self,
        graph_store: PropertyGraphStore,
        include_text: bool = True,
        include_properties: bool = False,
        synonym_prompt: Union[
            BasePromptTemplate, str
        ] = DEFAULT_SYNONYM_EXPAND_TEMPLATE,
        max_keywords: int = 10,
        path_depth: int = 1,
        limit: int = 30,
        output_parsing_fn: Optional[Callable] = None,
        llm: Optional[LLM] = None,
        **kwargs: Any,
    ) -> None:
        self._llm = llm or Settings.llm
        if isinstance(synonym_prompt, str):
            synonym_prompt = PromptTemplate(synonym_prompt)
        self._synonym_prompt = synonym_prompt
        self._output_parsing_fn = output_parsing_fn
        self._max_keywords = max_keywords
        self._path_depth = path_depth
        self._limit = limit
        super().__init__(
            graph_store=graph_store,
            include_text=include_text,
            include_properties=include_properties,
            **kwargs,
        )

    def _parse_llm_output(self, output: str) -> List[str]:
        if self._output_parsing_fn:
            matches = self._output_parsing_fn(output)
        else:
            matches = output.strip().split("^")

        # capitalize to normalize with ingestion
        return [x.strip().capitalize() for x in matches if x.strip()]

    def _prepare_matches(
        self, matches: List[str], limit: Optional[int] = None
    ) -> List[NodeWithScore]:
        kg_nodes = self._graph_store.get(ids=matches)
        triplets = self._graph_store.get_rel_map(
            kg_nodes,
            depth=self._path_depth,
            limit=limit or self._limit,
            ignore_rels=[KG_SOURCE_REL],
        )

        return self._get_nodes_with_score(triplets)

    async def _aprepare_matches(
        self, matches: List[str], limit: Optional[int] = None
    ) -> List[NodeWithScore]:
        kg_nodes = await self._graph_store.aget(ids=matches)
        triplets = await self._graph_store.aget_rel_map(
            kg_nodes,
            depth=self._path_depth,
            limit=limit or self._limit,
            ignore_rels=[KG_SOURCE_REL],
        )

        return self._get_nodes_with_score(triplets)

    def retrieve_from_graph(
        self, query_bundle: QueryBundle, limit: Optional[int] = None
    ) -> List[NodeWithScore]:
        response = self._llm.predict(
            self._synonym_prompt,
            query_str=query_bundle.query_str,
            max_keywords=self._max_keywords,
        )
        matches = self._parse_llm_output(response)

        return self._prepare_matches(matches, limit=limit or self._limit)

    async def aretrieve_from_graph(
        self, query_bundle: QueryBundle, limit: Optional[int] = None
    ) -> List[NodeWithScore]:
        response = await self._llm.apredict(
            self._synonym_prompt,
            query_str=query_bundle.query_str,
            max_keywords=self._max_keywords,
        )
        matches = self._parse_llm_output(response)

        return await self._aprepare_matches(matches, limit=limit or self._limit)
