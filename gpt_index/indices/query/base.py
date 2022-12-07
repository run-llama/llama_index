"""Base query classes."""

from abc import abstractmethod
from typing import Any, Generic, List, Optional, TypeVar

from gpt_index.indices.data_structs import IndexStruct
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.schema import DocumentStore, BaseQueryRunner

IS = TypeVar("IS", bound=IndexStruct)


class BaseGPTIndexQuery(Generic[IS]):
    """Base GPT Index Query.

    Helper class that is used to query an index. Can be called within `query`
    method of a BaseGPTIndex object, or instantiated independently.

    """

    def __init__(
        self, 
        index_struct: IS, 
        # TODO: pass from superclass
        llm_predictor: Optional[LLMPredictor] = None,
        docstore: Optional[DocumentStore] = None,
        query_runner: Optional[BaseQueryRunner] = None
    ) -> None:
        """Initialize with parameters."""
        if index_struct is None:
            raise ValueError("index_struct must be provided.")
        self._validate_index_struct(index_struct)
        self._index_struct = index_struct
        self._llm_predictor = llm_predictor or LLMPredictor()
        self._docstore = docstore
        self._query_runner = query_runner

    def _query_index_struct(self, query_str: str, index_struct: IndexStruct) -> List[Any]:
        """Recursively query the index struct."""
        if self._query_runner is None:
            raise ValueError("query_runner must be provided.")
        return self._query_runner.query(query_str, index_struct)

    @property
    def index_struct(self) -> IS:
        """Get the index struct."""
        return self._index_struct

    def _validate_index_struct(self, index_struct: IS) -> None:
        """Validate the index struct."""
        pass

    @abstractmethod
    def query(self, query_str: str, verbose: bool = False) -> str:
        """Answer a query."""

    def set_llm_predictor(self, llm_predictor: LLMPredictor) -> None:
        """Set LLM predictor."""
        self._llm_predictor = llm_predictor

