"""Query runner."""

from typing import Any, Dict, List, Optional, Union, cast

from gpt_index.data_structs.data_structs import IndexStruct
from gpt_index.docstore import DocumentStore
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.indices.prompt_helper import PromptHelper
from gpt_index.indices.query.base import BaseQueryRunner
from gpt_index.indices.query.query_transform import BaseQueryTransform
from gpt_index.indices.query.schema import QueryBundle, QueryConfig, QueryMode
from gpt_index.indices.registry import IndexRegistry
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.response.schema import Response

# TMP: refactor query config type
QUERY_CONFIG_TYPE = Union[Dict, QueryConfig]


class QueryRunner(BaseQueryRunner):
    """Tool to take in a query request and perform a query with the right classes.

    Higher-level wrapper over a given query.

    """

    def __init__(
        self,
        llm_predictor: LLMPredictor,
        prompt_helper: PromptHelper,
        embed_model: BaseEmbedding,
        docstore: DocumentStore,
        index_registry: IndexRegistry,
        query_configs: Optional[List[QUERY_CONFIG_TYPE]] = None,
        query_transform: Optional[BaseQueryTransform] = None,
        recursive: bool = False,
    ) -> None:
        """Init params."""
        config_dict: Dict[str, QueryConfig] = {}
        if query_configs is None or len(query_configs) == 0:
            query_config_objs: List[QueryConfig] = []
        elif isinstance(query_configs[0], Dict):
            query_config_objs = [
                QueryConfig.from_dict(cast(Dict, qc)) for qc in query_configs
            ]
        else:
            query_config_objs = [cast(QueryConfig, q) for q in query_configs]

        for qc in query_config_objs:
            config_dict[qc.index_struct_type] = qc

        self._config_dict = config_dict
        self._llm_predictor = llm_predictor
        self._prompt_helper = prompt_helper
        self._embed_model = embed_model
        self._docstore = docstore
        self._index_registry = index_registry
        self._query_transform = query_transform or BaseQueryTransform()
        self._recursive = recursive

    def _get_query_kwargs(self, config: QueryConfig) -> Dict[str, Any]:
        """Get query kwargs.

        Also update with default arguments if not present.

        """
        query_kwargs = {k: v for k, v in config.query_kwargs.items()}
        if "prompt_helper" not in query_kwargs:
            query_kwargs["prompt_helper"] = self._prompt_helper
        if "llm_predictor" not in query_kwargs:
            query_kwargs["llm_predictor"] = self._llm_predictor
        if "embed_model" not in query_kwargs:
            query_kwargs["embed_model"] = self._embed_model
        return query_kwargs

    def query(
        self, query_str_or_bundle: Union[str, QueryBundle], index_struct: IndexStruct
    ) -> Response:
        """Run query."""
        # NOTE: Currently, query transform is only run once
        # TODO: Consider refactor to support index-specific query transform
        if isinstance(query_str_or_bundle, str):
            query_bundle = self._query_transform(query_str_or_bundle)
        else:
            query_bundle = query_str_or_bundle

        index_struct_type = index_struct.get_type()
        if index_struct_type not in self._config_dict:
            config = QueryConfig(
                index_struct_type=index_struct_type, query_mode=QueryMode.DEFAULT
            )
        else:
            config = self._config_dict[index_struct_type]
        mode = config.query_mode

        query_cls = self._index_registry.type_to_query[index_struct_type][mode]
        # if recursive, pass self as query_runner to each individual query
        query_runner = self if self._recursive else None
        query_kwargs = self._get_query_kwargs(config)
        query_obj = query_cls(
            index_struct,
            **query_kwargs,
            query_runner=query_runner,
            docstore=self._docstore,
        )

        return query_obj.query(query_bundle)
