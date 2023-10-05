import logging
from typing import Any, List, Optional, cast

from llama_index.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.indices.vector_store.retrievers.auto_retriever.output_parser import (
    VectorStoreQueryOutputParser,
)
from llama_index.indices.vector_store.retrievers.auto_retriever.prompts import (
    DEFAULT_VECTOR_STORE_QUERY_PROMPT_TMPL,
    VectorStoreQueryPrompt,
)
from llama_index.output_parsers.base import OutputParserException, StructuredOutput
from llama_index.schema import NodeWithScore
from llama_index.vector_stores.types import (
    MetadataFilters,
    VectorStoreInfo,
    VectorStoreQueryMode,
    VectorStoreQuerySpec,
)

_logger = logging.getLogger(__name__)


class VectorIndexAutoRetriever(BaseRetriever):
    """Vector store auto retriever.

    A retriever for vector store index that uses an LLM to automatically set
    vector store query parameters.

    Args:
        index (VectorStoreIndex): vector store index
        vector_store_info (VectorStoreInfo): additional information information about
            vector store content and supported metadata filters. The natural language
            description is used by an LLM to automatically set vector store query
            parameters.
        prompt_template_str: custom prompt template string for LLM.
            Uses default template string if None.
        service_context: service context containing reference to LLMPredictor.
            Uses service context from index be default if None.
        similarity_top_k (int): number of top k results to return.
        max_top_k (int):
            the maximum top_k allowed. The top_k set by LLM or similarity_top_k will
            be clamped to this value.
        vector_store_query_mode (str): vector store query mode
            See reference for VectorStoreQueryMode for full list of supported modes.
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        vector_store_info: VectorStoreInfo,
        prompt_template_str: Optional[str] = None,
        service_context: Optional[ServiceContext] = None,
        max_top_k: int = 10,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        vector_store_query_mode: VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT,
        **kwargs: Any,
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
        self._similarity_top_k = similarity_top_k
        self._vector_store_query_mode = vector_store_query_mode
        self._kwargs = kwargs

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # prepare input
        info_str = self._vector_store_info.json(indent=4)
        schema_str = VectorStoreQuerySpec.schema_json(indent=4)

        # call LLM
        output = self._service_context.llm_predictor.predict(
            self._prompt,
            schema_str=schema_str,
            info_str=info_str,
            query_str=query_bundle.query_str,
        )

        # parse output
        assert self._prompt.output_parser is not None
        try:
            structured_output = cast(
                StructuredOutput, self._prompt.output_parser.parse(output)
            )
            query_spec = cast(VectorStoreQuerySpec, structured_output.parsed_output)
        except OutputParserException:
            _logger.warning("Failed to parse query spec, using defaults as fallback.")
            query_spec = VectorStoreQuerySpec(
                query=query_bundle.query_str,
                filters=[],
                top_k=None,
            )

        _logger.info(f"Using query str: {query_spec.query}")
        filter_dict = {filter.key: filter.value for filter in query_spec.filters}
        _logger.info(f"Using filters: {filter_dict}")

        if query_spec.top_k is None:
            similarity_top_k = self._similarity_top_k
        else:
            similarity_top_k = min(
                query_spec.top_k, self._max_top_k, self._similarity_top_k
            )

        _logger.info(f"Using top_k: {similarity_top_k}")

        retriever = VectorIndexRetriever(
            self._index,
            filters=MetadataFilters(filters=query_spec.filters),
            similarity_top_k=similarity_top_k,
            vector_store_query_mode=self._vector_store_query_mode,
            **self._kwargs,
        )
        return retriever.retrieve(query_spec.query)
