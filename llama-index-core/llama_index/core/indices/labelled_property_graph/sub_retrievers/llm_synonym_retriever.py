from typing import List, Optional

from llama_index.core.llms.llm import LLM
from llama_index.core.indices.keyword_table.utils import simple_extract_keywords
from llama_index.core.indices.labelled_property_graph.base import (
    LabelledPropertyGraphIndex,
)
from llama_index.core.indices.labelled_property_graph.sub_retrievers.base import (
    BaseLPGRetriever,
)
from llama_index.core.settings import Settings
from llama_index.core.schema import NodeWithScore, TextNode

DEFAULT_SYNONYM_EXPAND_TEMPLATE = (
    "Given some initial keywords, generate synonyms or related keywords up to {max_keywords} in total, "
    "considering possible cases of capitalization, pluralization, common expressions, etc.\n"
    "Provide all synonyms of keywords separated by '^' symbols: 'keyword1^keyword2^...'\n"
    "Note, result should be in one-line, separated by '^' symbols."
    "----\n"
    "KEYWORDS: {keywords}\n"
    "----\n"
    "SYNONYMS: "
)


class LLMSynonymRetriever(BaseLPGRetriever):
    def __init__(
        self,
        index: LabelledPropertyGraphIndex,
        include_text: bool = True,
        synonym_prompt_str: str = DEFAULT_SYNONYM_EXPAND_TEMPLATE,
        max_keywords: int = 10,
        output_parsing_fn: Optional[callable] = None,
        llm: Optional[LLM] = None,
        **kwargs,
    ) -> None:
        self._llm = llm or Settings.llm
        self._synonym_prompt_str = synonym_prompt_str
        self._output_parsing_fn = output_parsing_fn
        self._max_keywords = max_keywords
        super().__init__(index=index, include_text=include_text, **kwargs)

    def _parse_llm_output(self, output: str) -> List[str]:
        if self._output_parsing_fn:
            matches = self._output_parsing_fn(output)
        else:
            matches = output.strip().split("^")

        # capitalize to normalize with ingestion
        return [x.strip().capitalize() for x in matches if x.strip()]

    def _prepare_matches(self, matches: List[str]) -> List[NodeWithScore]:
        results = []
        for match in matches:
            sub_results = []
            sub_results.extend(
                self._storage_context.lpg_graph_store.get(entity_names=[match])
            )
            sub_results.extend(
                self._storage_context.lpg_graph_store.get(relation_names=[match])
            )

            for triplet in sub_results:
                id_ = triplet[0].properties.get("id_", None)
                assert id_ is not None

                text = f"{triplet[0].name}, {triplet[1].name}, {triplet[2].name}"
                results.append(
                    NodeWithScore(node=TextNode(id_=id_, text=text), score=1.0)
                )

        return results

    def _retrieve(self, query_bundle):
        keywords = simple_extract_keywords(query_bundle.query_str)

        response = self._llm.complete(
            self._synonym_prompt_str.format(
                keywords=keywords, max_keywords=self._max_keywords
            )
        )
        matches = self._parse_llm_output(response.text)

        results = self._prepare_matches(matches)

        return self._parse_results(results)

    async def _aretrieve(self, query_bundle):
        keywords = simple_extract_keywords(query_bundle.query_str)

        response = await self._llm.acomplete(
            self._synonym_prompt_str.format(
                keywords=keywords, max_keywords=self._max_keywords
            )
        )
        matches = self._parse_llm_output(response.text)

        results = self._prepare_matches(matches)

        return self._parse_results(results)
