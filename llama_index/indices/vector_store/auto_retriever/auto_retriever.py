


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
from llama_index.vector_stores.types import MetadataFilters, VectorStoreQuery


class VectorIndexAutoRetriever(base_retriever):
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
        info_str = self._vector_store_info.to_json()

        # call LLM to get output
        output = self._service_context.llm_predictor.predict(
            self._prompt, 
            info_str=info_str,
            query_str=query_bundle.query_str
        )

        parse = self._prompt.output_parser.parse(output)

        # parse vector store query from output
        filters = cast(MetadataFilters, parse.parsed_output)

        query = VectorStoreQuery(
            query_str=query_bundle.query_str,
            filters=filters,
       )
    
        # query vector store
        query_result = self._vector_store.query(query)

        # parse into final result
        ...

        return node_with_scores