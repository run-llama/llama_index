"""Query runner."""

from gpt_index.schema import BaseQueryRunner
from gpt_index.indices.data_structs import Node, IndexStruct, IndexStructType
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin

from gpt_index.indices.query.query_map import get_query_cls
from gpt_index.indices.query.schema import QueryMode

from typing import List, Dict, Any, Optional
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.indices.query.schema import QueryConfig


class QueryRunner(BaseQueryRunner):
    """Tool to take in a query request and perform a query with the right classes.

    Higher-level wrapper over a given query.

    """

    def __init__(
        self,
        query_configs: List[Dict], 
        llm_predictor: Optional[LLMPredictor] = None,
        verbose: bool = False
    ) -> None:
        """Init params."""
        config_dict: Dict[IndexStructType, QueryConfig] = {}
        for qc_dict in query_configs:
            qc = QueryConfig.from_dict(qc_dict)
            config_dict[qc.index_struct_type] = qc
        self._config_dict = config_dict
        self._llm_predictor = llm_predictor
        self._verbose = verbose

    def query(self, query_str: str, index_struct: IndexStruct) -> str:
        """Run query."""
        index_struct_type = IndexStructType.from_index_struct(index_struct)
        if index_struct_type not in self._config_dict:
            raise ValueError(f"IndexStructType {index_struct_type} not in config_dict")
        config = self._config_dict[index_struct_type]
        mode = config.query_mode
        query_cls = get_query_cls(index_struct_type, mode)
        query_obj = query_cls(index_struct, **config.query_kwargs, query_runner=self)

        # set llm_predictor if exists
        query_obj.set_llm_predictor(self._llm_predictor)

        return query_obj.query(query_str, verbose=self._verbose)