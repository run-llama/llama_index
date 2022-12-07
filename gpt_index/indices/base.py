"""Base index classes."""
import json
from abc import abstractmethod
from typing import Any, Generic, List, Optional, Sequence, TypeVar, cast

from gpt_index.indices.data_structs import IndexStruct
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.query_runner import QueryRunner
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.schema import BaseDocument, DocumentStore

IS = TypeVar("IS", bound=IndexStruct)

DEFAULT_MODE = "default"
EMBEDDING_MODE = "embedding"


class BaseGPTIndex(Generic[IS]):
    """Base GPT Index."""

    def __init__(
        self,
        documents: Optional[Sequence[BaseDocument]] = None,
        index_struct: Optional[IS] = None,
        llm_predictor: Optional[LLMPredictor] = None,
    ) -> None:
        """Initialize with parameters."""
        if index_struct is None and documents is None:
            raise ValueError("One of documents or index_struct must be provided.")
        if index_struct is not None and documents is not None:
            raise ValueError("Only one of documents or index_struct can be provided.")

        self._llm_predictor = llm_predictor or LLMPredictor()

        # build index struct in the init function
        if index_struct is not None:
            self._index_struct = index_struct
        else:
            documents = cast(List[BaseDocument], documents)
            # TODO: introduce document store outside __init__ function
            self._docstore = DocumentStore.from_documents(documents)
            self._index_struct = self.build_index_from_documents(documents)

    @property
    def index_struct(self) -> IS:
        """Get the index struct."""
        return self._index_struct

    @property
    def index_struct_with_text(self) -> IS:
        """Get the index struct with text.

        If text not set, raise an error.
        For use when composing indices with other indices.

        """
        # make sure that we generate text for index struct
        if self._index_struct.text is None:
            raise ValueError(
                "Index must have text property set in order "
                "to be composed with other indices. "
                "In order to set text, please run `index.set_text()`."
            )
        return self._index_struct

    def set_text(self, text: str) -> None:
        """Set text for index struct.

        This allows index_struct_with_text to be used to compose indices
        with other indices.

        """
        self._index_struct.text = text

    @abstractmethod
    def build_index_from_documents(self, documents: Sequence[BaseDocument]) -> IS:
        """Build the index from documents."""

    @abstractmethod
    def insert(self, document: BaseDocument, **insert_kwargs: Any) -> None:
        """Insert a document."""

    @abstractmethod
    def delete(self, document: BaseDocument) -> None:
        """Delete a document."""

    @abstractmethod
    def _mode_to_query(self, mode: str, **query_kwargs: Any) -> BaseGPTIndexQuery:
        """Query mode to class."""

    def query(
        self,
        query_str: str,
        verbose: bool = False,
        mode: str = DEFAULT_MODE,
        **query_kwargs: Any
    ) -> str:
        """Answer a query."""
        # TODO: remove _mode_to_query and consolidate with query_runner
        if mode == "recursive":
            if "query_configs" not in query_kwargs:
                raise ValueError("query_configs must be provided for recursive mode.")
            query_configs = query_kwargs["query_configs"]
            query_runner = QueryRunner(
                query_configs, self._llm_predictor, verbose=verbose
            )
            return query_runner.query(query_str, self._index_struct)
        else:
            query_obj = self._mode_to_query(mode, **query_kwargs)
            # set llm_predictor if exists
            query_obj.set_llm_predictor(self._llm_predictor)
            return query_obj.query(query_str, verbose=verbose)

    @classmethod
    @abstractmethod
    def load_from_disk(cls, save_path: str, **kwargs: Any) -> "BaseGPTIndex":
        """Load from disk."""

    def save_to_disk(self, save_path: str) -> None:
        """Safe to file."""
        with open(save_path, "w") as f:
            json.dump(self.index_struct.to_dict(), f)
