


import logging
from typing import Any, Dict, List, cast

from llama_index.data_structs.node import NodeWithScore
from llama_index.indices import base_retriever
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.vector_store.auto_retriever.output_parser import \
    VectorStoreQueryOutputParser
from llama_index.indices.vector_store.auto_retriever.prompts import (
    DEFAULT_VECTOR_STORE_QUERY_PROMPT_TMPL, VectorStoreQueryPrompt)
from llama_index.indices.vector_store.auto_retriever.schema import \
    VectorStoreInfo
from llama_index.indices.vector_store.base import GPTVectorStoreIndex
from llama_index.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.output_parsers.base import StructuredOutput
from llama_index.vector_stores.types import (MetadataFilters,
                                             VectorStoreQuerySpec)

_logger = logging.getLogger(__name__)

class VectorIndexAutoRetriever(base_retriever.BaseRetriever):
    def __init__(
            self,
            index: GPTVectorStoreIndex,
            vector_store_info: VectorStoreInfo,
            **kwargs: Any,
        ) -> None:
        self._index = index
        self._vector_store = self._index.vector_store
        self._service_context = self._index.service_context
        self._docstore = self._index.docstore

        self._kwargs: Dict[str, Any] = kwargs.get("retriever_kwargs", {})

        self._vector_store_info = vector_store_info
        self._prompt =  VectorStoreQueryPrompt(
            template=DEFAULT_VECTOR_STORE_QUERY_PROMPT_TMPL,
            output_parser=VectorStoreQueryOutputParser(),
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # prepare input
        info_str = self._vector_store_info.to_json(indent=4)
        schema_str = VectorStoreQuerySpec.schema_json(indent=4)

        # call LLM
        output, _ = self._service_context.llm_predictor.predict(
            self._prompt, 
            schema_str=schema_str,
            info_str=info_str,
            query_str=query_bundle.query_str
        )

        # parse output
        structured_output = cast(StructuredOutput, self._prompt.output_parser.parse(output))
        query_and_filters = cast(VectorStoreQuerySpec, structured_output.parsed_output)

        _logger.info(f'Auto query: {query_and_filters.query}')
        filter_dict = {
            filter.key: filter.value for filter in query_and_filters.filters
        }
        _logger.info(f'Auto filter: {filter_dict}')
        
        retriever = VectorIndexRetriever(self._index, filters=MetadataFilters(filters=query_and_filters.filters))
        return retriever.retrieve(query_and_filters.query)