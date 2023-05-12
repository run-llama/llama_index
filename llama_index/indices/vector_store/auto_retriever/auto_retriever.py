import logging
from typing import List, Optional, cast

from llama_index.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.data_structs.node import NodeWithScore
from llama_index.indices import base_retriever
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.vector_store.auto_retriever.output_parser import (
    VectorStoreQueryOutputParser,
)
from llama_index.indices.vector_store.auto_retriever.prompts import (
    DEFAULT_VECTOR_STORE_QUERY_PROMPT_TMPL,
    VectorStoreQueryPrompt,
)
from llama_index.indices.vector_store.auto_retriever.schema import VectorStoreInfo
from llama_index.indices.vector_store.base import GPTVectorStoreIndex
from llama_index.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.output_parsers.base import StructuredOutput
from llama_index.vector_stores.types import MetadataFilters, VectorStoreQuerySpec

_logger = logging.getLogger(__name__)


class VectorIndexAutoRetriever(base_retriever.BaseRetriever):
    def __init__(
        self,
        index: GPTVectorStoreIndex,
        vector_store_info: VectorStoreInfo,
        prompt_template_str: Optional[str] = None,
        service_context: Optional[ServiceContext] = None,
        max_top_k: int = 10,
    ) -> None:
        self._index = index
        self._vector_store_info = vector_store_info
        self._service_context = service_context or self._index.service_context

        # prompt
        prompt_template_str = (
            prompt_template_str or DEFAULT_VECTOR_STORE_QUERY_PROMPT_TMPL
        )
        output_parser = VectorStoreQueryOutputParser()
        self._prompt = VectorStoreQueryPrompt(
            template=prompt_template_str,
            output_parser=output_parser,
        )

        # additional config
        self._max_top_k = max_top_k

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # prepare input
        info_str = self._vector_store_info.json(indent=4)
        schema_str = VectorStoreQuerySpec.schema_json(indent=4)

        # call LLM
        output, _ = self._service_context.llm_predictor.predict(
            self._prompt,
            schema_str=schema_str,
            info_str=info_str,
            query_str=query_bundle.query_str,
        )

        # parse output
        assert self._prompt.output_parser is not None
        structured_output = cast(
            StructuredOutput, self._prompt.output_parser.parse(output)
        )
        query_spec = cast(VectorStoreQuerySpec, structured_output.parsed_output)

        _logger.info(f"Auto query: {query_spec.query}")
        filter_dict = {filter.key: filter.value for filter in query_spec.filters}
        _logger.info(f"Auto filter: {filter_dict}")

        if query_spec.top_k is None:
            similarity_top_k = DEFAULT_SIMILARITY_TOP_K
        else:
            similarity_top_k = min(query_spec.top_k, self._max_top_k)

        _logger.info(f"Auto top_k: {similarity_top_k}")

        retriever = VectorIndexRetriever(
            self._index,
            filters=MetadataFilters(filters=query_spec.filters),
            similarity_top_k=similarity_top_k,
        )
        return retriever.retrieve(query_spec.query)
