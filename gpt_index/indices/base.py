"""Base index classes."""
import json
from abc import abstractmethod
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
)

from gpt_index.indices.data_structs import IndexStruct, IndexStructType
from gpt_index.indices.prompt_helper import PromptHelper
from gpt_index.indices.query.query_runner import QueryRunner
from gpt_index.indices.query.schema import QueryConfig, QueryMode
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.schema import BaseDocument, DocumentStore
from gpt_index.utils import llm_token_counter

IS = TypeVar("IS", bound=IndexStruct)


DOCUMENTS_INPUT = Union[BaseDocument, "BaseGPTIndex"]


class BaseGPTIndex(Generic[IS]):
    """Base GPT Index."""

    index_struct_cls: Type[IS]

    def __init__(
        self,
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        index_struct: Optional[IS] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        docstore: Optional[DocumentStore] = None,
        prompt_helper: Optional[PromptHelper] = None,
        chunk_size_limit: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize with parameters."""
        if index_struct is None and documents is None:
            raise ValueError("One of documents or index_struct must be provided.")
        if index_struct is not None and documents is not None:
            raise ValueError("Only one of documents or index_struct can be provided.")

        self._llm_predictor = llm_predictor or LLMPredictor()

        # TODO: move out of base if we need custom params per index
        self._prompt_helper = prompt_helper or PromptHelper.from_llm_predictor(
            self._llm_predictor, chunk_size_limit=chunk_size_limit
        )

        # build index struct in the init function
        self._docstore = docstore or DocumentStore()
        if index_struct is not None:
            self._index_struct = index_struct
        else:
            documents = cast(Sequence[DOCUMENTS_INPUT], documents)
            documents = self._process_documents(documents, self._docstore)
            self._validate_documents(documents)
            # TODO: introduce document store outside __init__ function
            self._index_struct = self.build_index_from_documents(
                documents, verbose=verbose
            )

    @property
    def docstore(self) -> DocumentStore:
        """Get the docstore corresponding to the index."""
        return self._docstore

    def _process_documents(
        self, documents: Sequence[DOCUMENTS_INPUT], docstore: DocumentStore
    ) -> List[BaseDocument]:
        """Process documents."""
        results = []
        for doc in documents:
            if isinstance(doc, BaseGPTIndex):
                # if user passed in another index, we need to extract the index struct,
                # and also update docstore with the docstore in the index
                results.append(doc.index_struct_with_text)
                docstore.update_docstore(doc.docstore)
            else:
                results.append(doc)
        docstore.add_documents(results)
        return results

    def _validate_documents(self, documents: Sequence[BaseDocument]) -> None:
        """Validate documents."""
        for doc in documents:
            if not isinstance(doc, BaseDocument):
                raise ValueError("Documents must be of type BaseDocument.")

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
    def _build_index_from_documents(
        self, documents: Sequence[BaseDocument], verbose: bool = False
    ) -> IS:
        """Build the index from documents."""

    @llm_token_counter("build_index_from_documents")
    def build_index_from_documents(
        self, documents: Sequence[BaseDocument], verbose: bool = False
    ) -> IS:
        """Build the index from documents."""
        return self._build_index_from_documents(documents, verbose=verbose)

    @abstractmethod
    def _insert(self, document: BaseDocument, **insert_kwargs: Any) -> None:
        """Insert a document."""

    @llm_token_counter("insert")
    def insert(self, document: DOCUMENTS_INPUT, **insert_kwargs: Any) -> None:
        """Insert a document."""
        processed_doc = self._process_documents([document], self._docstore)[0]
        self._validate_documents([processed_doc])
        self._insert(processed_doc, **insert_kwargs)

    @abstractmethod
    def delete(self, document: BaseDocument) -> None:
        """Delete a document."""

    def _preprocess_query(self, mode: QueryMode, query_kwargs: Dict) -> None:
        """Preprocess query.

        This allows subclasses to pass in additional query kwargs
        to query, for instance arguments that are shared between the
        index and the query class. By default, this does nothing.
        This also allows subclasses to do validation.

        """
        pass

    def query(
        self,
        query_str: str,
        verbose: bool = False,
        mode: str = QueryMode.DEFAULT,
        **query_kwargs: Any
    ) -> str:
        """Answer a query.

        When `query` is called, we query the index with the given `mode` and
        `query_kwargs`. The `mode` determines the type of query to run, and
        `query_kwargs` are parameters that are specific to the query type.

        For a comprehensive documentation of available `mode` and `query_kwargs` to
        query a given index, please visit :ref:`Ref-Query`.


        """
        mode_enum = QueryMode(mode)
        # TODO: remove _mode_to_query and consolidate with query_runner
        if mode_enum == QueryMode.RECURSIVE:
            if "query_configs" not in query_kwargs:
                raise ValueError("query_configs must be provided for recursive mode.")
            query_configs = query_kwargs["query_configs"]
            query_runner = QueryRunner(
                self._llm_predictor,
                self._prompt_helper,
                self._docstore,
                query_configs=query_configs,
                verbose=verbose,
                recursive=True,
            )
            return query_runner.query(query_str, self._index_struct)
        else:
            self._preprocess_query(mode_enum, query_kwargs)
            # TODO: pass in query config directly
            query_config = QueryConfig(
                index_struct_type=IndexStructType.from_index_struct(self._index_struct),
                query_mode=mode_enum,
                query_kwargs=query_kwargs,
            ).to_dict()
            query_runner = QueryRunner(
                self._llm_predictor,
                self._prompt_helper,
                self._docstore,
                query_configs=[query_config],
                verbose=verbose,
                recursive=False,
            )
            return query_runner.query(query_str, self._index_struct)

    @classmethod
    def load_from_disk(cls, save_path: str, **kwargs: Any) -> "BaseGPTIndex":
        """Load index from disk.

        This method loads the index from a JSON file stored on disk. The index data
        structure itself is preserved completely. If the index is defined over
        subindices, those subindices will also be preserved (and subindices of
        those subindices, etc.).

        Args:
            save_path (str): The save_path of the file.

        Returns:
            BaseGPTIndex: The loaded index.

        """
        with open(save_path, "r") as f:
            result_dict = json.load(f)
            index_struct = cls.index_struct_cls.from_dict(result_dict["index_struct"])
            docstore = DocumentStore.from_dict(result_dict["docstore"])
            return cls(index_struct=index_struct, docstore=docstore, **kwargs)

    def save_to_disk(self, save_path: str, **save_kwargs: Any) -> None:
        """Save to file.

        This method stores the index into a JSON file stored on disk.

        Args:
            save_path (str): The save_path of the file.

        """
        out_dict: Dict[str, dict] = {
            "index_struct": self.index_struct.to_dict(),
            "docstore": self.docstore.to_dict(),
        }
        with open(save_path, "w") as f:
            json.dump(out_dict, f)
