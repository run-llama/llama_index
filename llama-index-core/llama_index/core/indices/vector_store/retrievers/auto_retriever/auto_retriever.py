import logging
from typing import Any, List, Optional, Tuple, cast

from llama_index.core.base.base_auto_retriever import BaseAutoRetriever
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.indices.vector_store.retrievers.auto_retriever.output_parser import (
    VectorStoreQueryOutputParser,
)
from llama_index.core.indices.vector_store.retrievers.auto_retriever.prompts import (
    DEFAULT_VECTOR_STORE_QUERY_PROMPT_TMPL,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.output_parsers.base import (
    OutputParserException,
    StructuredOutput,
)
from llama_index.core.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.schema import IndexNode, QueryBundle
from llama_index.core.settings import Settings
from llama_index.core.vector_stores.types import (
    FilterCondition,
    MetadataFilters,
    VectorStoreInfo,
    VectorStoreQueryMode,
    VectorStoreQuerySpec,
)

_logger = logging.getLogger(__name__)


class VectorIndexAutoRetriever(BaseAutoRetriever):
    """
    Vector store auto retriever.

    A retriever for vector store index that uses an LLM to automatically set
    vector store query parameters.

    Args:
        index (VectorStoreIndex): vector store index
        vector_store_info (VectorStoreInfo): additional information about
            vector store content and supported metadata filters. The natural language
            description is used by an LLM to automatically set vector store query
            parameters.
        prompt_template_str: custom prompt template string for LLM.
            Uses default template string if None.
        similarity_top_k (int): number of top k results to return.
        empty_query_top_k (Optional[int]): number of top k results to return
            if the inferred query string is blank (uses metadata filters only).
            Can be set to None, which would use the similarity_top_k instead.
            By default, set to 10.
        max_top_k (int):
            the maximum top_k allowed. The top_k set by LLM or similarity_top_k will
            be clamped to this value.
        vector_store_query_mode (str): vector store query mode
            See reference for VectorStoreQueryMode for full list of supported modes.
        default_empty_query_vector (Optional[List[float]]): default empty query vector.
            Defaults to None. If not None, then this vector will be used as the query
            vector if the query is empty.
        callback_manager (Optional[CallbackManager]): callback manager
        verbose (bool): verbose mode
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        vector_store_info: VectorStoreInfo,
        llm: Optional[LLM] = None,
        prompt_template_str: Optional[str] = None,
        max_top_k: int = 10,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        empty_query_top_k: Optional[int] = 10,
        vector_store_query_mode: VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT,
        default_empty_query_vector: Optional[List[float]] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        extra_filters: Optional[MetadataFilters] = None,
        object_map: Optional[dict] = None,
        objects: Optional[List[IndexNode]] = None,
        **kwargs: Any,
    ) -> None:
        self._index = index
        self._vector_store_info = vector_store_info
        self._default_empty_query_vector = default_empty_query_vector
        self._llm = llm or Settings.llm
        callback_manager = callback_manager or Settings.callback_manager

        # prompt
        prompt_template_str = (
            prompt_template_str or DEFAULT_VECTOR_STORE_QUERY_PROMPT_TMPL
        )
        self._output_parser = VectorStoreQueryOutputParser()
        self._prompt: BasePromptTemplate = PromptTemplate(template=prompt_template_str)

        # additional config
        self._max_top_k = max_top_k
        self._similarity_top_k = similarity_top_k
        self._empty_query_top_k = empty_query_top_k
        self._vector_store_query_mode = vector_store_query_mode
        # if extra_filters is OR condition, we don't support that yet
        if extra_filters is not None and extra_filters.condition == FilterCondition.OR:
            raise ValueError("extra_filters cannot be OR condition")
        self._extra_filters = extra_filters or MetadataFilters(filters=[])
        self._kwargs = kwargs
        super().__init__(
            callback_manager=callback_manager,
            object_map=object_map or self._index._object_map,
            objects=objects,
            verbose=verbose,
        )

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {
            "prompt": self._prompt,
        }

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Get prompt modules."""
        if "prompt" in prompts:
            self._prompt = prompts["prompt"]

    def _get_query_bundle(self, query: str) -> QueryBundle:
        """Get query bundle."""
        if not query and self._default_empty_query_vector is not None:
            return QueryBundle(
                query_str="",
                embedding=self._default_empty_query_vector,
            )
        else:
            return QueryBundle(query_str=query)

    def _parse_generated_spec(
        self, output: str, query_bundle: QueryBundle
    ) -> BaseModel:
        """Parse generated spec."""
        try:
            structured_output = cast(
                StructuredOutput, self._output_parser.parse(output)
            )
            query_spec = cast(VectorStoreQuerySpec, structured_output.parsed_output)
        except OutputParserException:
            _logger.warning("Failed to parse query spec, using defaults as fallback.")
            query_spec = VectorStoreQuerySpec(
                query=query_bundle.query_str,
                filters=[],
                top_k=None,
            )

        return query_spec

    def generate_retrieval_spec(
        self, query_bundle: QueryBundle, **kwargs: Any
    ) -> BaseModel:
        # prepare input
        info_str = self._vector_store_info.model_dump_json(indent=4)
        schema_str = VectorStoreQuerySpec.model_json_schema()

        # call LLM
        output = self._llm.predict(
            self._prompt,
            schema_str=schema_str,
            info_str=info_str,
            query_str=query_bundle.query_str,
        )

        # parse output
        return self._parse_generated_spec(output, query_bundle)

    async def agenerate_retrieval_spec(
        self, query_bundle: QueryBundle, **kwargs: Any
    ) -> BaseModel:
        # prepare input
        info_str = self._vector_store_info.model_dump_json(indent=4)
        schema_str = VectorStoreQuerySpec.model_json_schema()

        # call LLM
        output = await self._llm.apredict(
            self._prompt,
            schema_str=schema_str,
            info_str=info_str,
            query_str=query_bundle.query_str,
        )

        # parse output
        return self._parse_generated_spec(output, query_bundle)

    def _build_retriever_from_spec(  # type: ignore
        self, spec: VectorStoreQuerySpec
    ) -> Tuple[BaseRetriever, QueryBundle]:
        # construct new query bundle from query_spec
        # insert 0 vector if query is empty and default_empty_query_vector is not None
        new_query_bundle = self._get_query_bundle(spec.query)

        _logger.info(f"Using query str: {spec.query}")
        filter_list = [
            (filter.key, filter.operator.value, filter.value) for filter in spec.filters
        ]
        _logger.info(f"Using filters: {filter_list}")
        if self._verbose:
            print(f"Using query str: {spec.query}")
            print(f"Using filters: {filter_list}")

        # define similarity_top_k
        # if query is specified, then use similarity_top_k
        # if query is blank, then use empty_query_top_k
        if spec.query or self._empty_query_top_k is None:
            similarity_top_k = self._similarity_top_k
        else:
            similarity_top_k = self._empty_query_top_k

        # if query_spec.top_k is specified, then use it
        # as long as below max_top_k and similarity_top_k
        if spec.top_k is not None:
            similarity_top_k = min(spec.top_k, self._max_top_k, similarity_top_k)

        _logger.info(f"Using top_k: {similarity_top_k}")

        # avoid passing empty filters to retriever
        if len(spec.filters) + len(self._extra_filters.filters) == 0:
            filters = None
        else:
            filters = MetadataFilters(
                filters=[*spec.filters, *self._extra_filters.filters]
            )

        return (
            VectorIndexRetriever(
                self._index,
                filters=filters,
                similarity_top_k=similarity_top_k,
                vector_store_query_mode=self._vector_store_query_mode,
                object_map=self.object_map,
                verbose=self._verbose,
                **self._kwargs,
            ),
            new_query_bundle,
        )
