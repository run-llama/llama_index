"""Query runner."""

from typing import Dict, List, Optional

from gpt_index.indices.data_structs import IndexStruct, IndexStructType
from gpt_index.indices.prompt_helper import PromptHelper
from gpt_index.indices.query.base import BaseQueryRunner
from gpt_index.indices.query.query_map import get_query_cls
from gpt_index.indices.query.schema import QueryConfig, QueryMode
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
]


class QueryRunner(BaseQueryRunner):
    """Tool to take in a query request and perform a query with the right classes.

    Higher-level wrapper over a given query.

    """

    def __init__(
        self,
        llm_predictor: LLMPredictor,
        prompt_helper: PromptHelper,
        docstore: DocumentStore,
        query_configs: Optional[List[Dict]] = None,
        verbose: bool = False,
        recursive: bool = False,
    ) -> None:
        """Init params."""
        config_dict: Dict[IndexStructType, QueryConfig] = {}
        if query_configs is None:
            query_config_objs = DEFAULT_QUERY_CONFIGS
        else:
            query_config_objs = [QueryConfig.from_dict(qc) for qc in query_configs]

        for qc in query_config_objs:
            config_dict[qc.index_struct_type] = qc
        self._config_dict = config_dict
        self._llm_predictor = llm_predictor
        self._prompt_helper = prompt_helper
        self._docstore = docstore
        self._verbose = verbose
        self._recursive = recursive

    def query(self, query_str: str, index_struct: IndexStruct) -> str:
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
