"""Query runner."""

from typing import Dict, List, Optional, Union, cast

from gpt_index.data_structs.data_structs import IndexStruct
from gpt_index.data_structs.struct_type import IndexStructType
from gpt_index.indices.prompt_helper import PromptHelper
from gpt_index.indices.query.base import BaseQueryRunner
from gpt_index.indices.query.query_map import get_query_cls
from gpt_index.indices.query.schema import QueryConfig, QueryMode
from gpt_index.indices.response.schema import Response
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.schema import DocumentStore

DEFAULT_QUERY_CONFIGS = [
    QueryConfig(
        index_struct_type=IndexStructType.TREE,
        query_mode=QueryMode.DEFAULT,
    ),
    QueryConfig(
        index_struct_type=IndexStructType.LIST,
        query_mode=QueryMode.DEFAULT,
    ),
    QueryConfig(
        index_struct_type=IndexStructType.KEYWORD_TABLE,
        query_mode=QueryMode.DEFAULT,
    ),
    QueryConfig(index_struct_type=IndexStructType.DICT, query_mode=QueryMode.DEFAULT),
    QueryConfig(
        index_struct_type=IndexStructType.SIMPLE_DICT, query_mode=QueryMode.DEFAULT
    ),
]

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
        docstore: DocumentStore,
        query_configs: Optional[List[QUERY_CONFIG_TYPE]] = None,
        verbose: bool = False,
        recursive: bool = False,
    ) -> None:
        """Init params."""
        config_dict: Dict[IndexStructType, QueryConfig] = {}
        if query_configs is None or len(query_configs) == 0:
            query_config_objs: List[QueryConfig] = DEFAULT_QUERY_CONFIGS
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
        self._docstore = docstore
        self._verbose = verbose
        self._recursive = recursive

    def query(self, query_str: str, index_struct: IndexStruct) -> Response:
        """Run query."""
        index_struct_type = IndexStructType.from_index_struct(index_struct)
        if index_struct_type not in self._config_dict:
            raise ValueError(f"IndexStructType {index_struct_type} not in config_dict")
        config = self._config_dict[index_struct_type]
        mode = config.query_mode
        query_cls = get_query_cls(index_struct_type, mode)
        # if recursive, pass self as query_runner to each individual query
        query_runner = self if self._recursive else None
        query_obj = query_cls(
            index_struct,
            prompt_helper=self._prompt_helper,
            **config.query_kwargs,
            query_runner=query_runner,
            docstore=self._docstore,
        )

        # TODO: refactor this, so we can pass in during init
        # set llm_predictor if exists
        if not query_obj._llm_predictor_set:
            query_obj.set_llm_predictor(self._llm_predictor)

        return query_obj.query(query_str, verbose=self._verbose)
