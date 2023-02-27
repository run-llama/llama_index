"""Base index classes."""
import json
import logging
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

from gpt_index.data_structs.data_structs import IndexStruct, Node
from gpt_index.docstore import DOC_TYPE, DocumentStore
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.node_utils import get_nodes_from_document
from gpt_index.indices.prompt_helper import PromptHelper
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.query_runner import QueryRunner
from gpt_index.indices.query.query_transform import BaseQueryTransform
from gpt_index.indices.query.schema import QueryBundle, QueryConfig, QueryMode
from gpt_index.indices.registry import IndexRegistry
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.text_splitter import TextSplitter, TokenTextSplitter
from gpt_index.readers.schema.base import Document
from gpt_index.response.schema import Response
from gpt_index.schema import BaseDocument
from gpt_index.token_counter.token_counter import llm_token_counter

IS = TypeVar("IS", bound=IndexStruct)


DOCUMENTS_INPUT = Union[BaseDocument, "BaseGPTIndex"]


class BaseGPTIndex(Generic[IS]):
    """Base LlamaIndex.

    Args:
        documents (Optional[Sequence[BaseDocument]]): List of documents to
            build the index from.
        llm_predictor (LLMPredictor): Optional LLMPredictor object. If not provided,
            will use the default LLMPredictor (text-davinci-003)
        prompt_helper (PromptHelper): Optional PromptHelper object. If not provided,
            will use the default PromptHelper.
        chunk_size_limit (Optional[int]): Optional chunk size limit. If not provided,
            will use the default chunk size limit (4096 max input size).
        include_extra_info (bool): Optional bool. If True, extra info (i.e. metadata)
            of each Document will be prepended to its text to help with queries.
            Default is True.

    """

    index_struct_cls: Type[IS]

    def __init__(
        self,
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        index_struct: Optional[IS] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        embed_model: Optional[BaseEmbedding] = None,
        docstore: Optional[DocumentStore] = None,
        index_registry: Optional[IndexRegistry] = None,
        prompt_helper: Optional[PromptHelper] = None,
        text_splitter: Optional[TextSplitter] = None,
        chunk_size_limit: Optional[int] = None,
        include_extra_info: bool = True,
    ) -> None:
        """Initialize with parameters."""
        if index_struct is None and documents is None:
            raise ValueError("One of documents or index_struct must be provided.")
        if index_struct is not None and documents is not None:
            raise ValueError("Only one of documents or index_struct can be provided.")

        self._llm_predictor = llm_predictor or LLMPredictor()
        # NOTE: the embed_model isn't used in all indices
        self._embed_model = embed_model or OpenAIEmbedding()
        self._include_extra_info = include_extra_info

        # TODO: move out of base if we need custom params per index
        self._prompt_helper = prompt_helper or PromptHelper.from_llm_predictor(
            self._llm_predictor, chunk_size_limit=chunk_size_limit
        )
        self._text_splitter = text_splitter or self._build_fallback_text_splitter()

        # build index struct in the init function
        self._docstore = docstore or DocumentStore()
        self._index_registry = index_registry or IndexRegistry()

        if index_struct is not None:
            if not isinstance(index_struct, self.index_struct_cls):
                raise ValueError(
                    f"index_struct must be of type {self.index_struct_cls}"
                )
            self._index_struct = index_struct
        else:
            documents = cast(Sequence[DOCUMENTS_INPUT], documents)
            documents = self._process_documents(
                documents, self._docstore, self._index_registry
            )
            self._validate_documents(documents)
            # TODO: introduce document store outside __init__ function
            self._index_struct = self.build_index_from_documents(documents)
        # update index registry and docstore with index_struct
        self._update_index_registry_and_docstore()

    @property
    def prompt_helper(self) -> PromptHelper:
        """Get the prompt helper corresponding to the index."""
        return self._prompt_helper

    @property
    def docstore(self) -> DocumentStore:
        """Get the docstore corresponding to the index."""
        return self._docstore

    @property
    def index_registry(self) -> IndexRegistry:
        """Get the index registry corresponding to the index."""
        return self._index_registry

    @property
    def llm_predictor(self) -> LLMPredictor:
        """Get the llm predictor."""
        return self._llm_predictor

    @property
    def embed_model(self) -> BaseEmbedding:
        """Get the llm predictor."""
        return self._embed_model

    def _update_index_registry_and_docstore(self) -> None:
        """Update index registry and docstore."""
        # update index registry with current struct
        cur_type = self.index_struct_cls.get_type()
        self._index_registry.type_to_struct[cur_type] = self.index_struct_cls
        self._index_registry.type_to_query[cur_type] = self.get_query_map()

        # update docstore with current struct
        # NOTE: we call allow_update=True: in old versions of the docstore,
        # the index_struct was not stored in the docstore. whereas
        # in the new docstore, index_struct is stored in the docstore.
        # if we want to break BW compatibility, we can just remove this line
        # and only insert into docstore during index construction.
        self._docstore.add_documents([self.index_struct], allow_update=True)

    def _process_documents(
        self,
        documents: Sequence[DOCUMENTS_INPUT],
        docstore: DocumentStore,
        index_registry: IndexRegistry,
    ) -> List[BaseDocument]:
        """Process documents."""
        results: List[DOC_TYPE] = []
        for doc in documents:
            if isinstance(doc, BaseGPTIndex):
                # if user passed in another index, we need to do the following:
                # - update docstore with the docstore in the index
                # - validate that the index is in the docstore
                # - update the index registry

                index_registry.update(doc.index_registry)
                docstore.update_docstore(doc.docstore)
                # assert that the doc exists within the docstore
                sub_index_struct = doc.index_struct_with_text
                if not docstore.document_exists(sub_index_struct.get_doc_id()):
                    raise ValueError(
                        "The index struct of the sub-index must exist in the docstore. "
                        f"Invalid doc ID: {sub_index_struct.get_doc_id()}"
                    )
                results.append(sub_index_struct)
            elif isinstance(doc, Document):
                results.append(doc)
            else:
                raise ValueError(f"Invalid document type: {type(doc)}.")
        return cast(List[BaseDocument], results)

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
            # NOTE: set text to be empty string for now
            raise ValueError(
                "Index must have text property set in order "
                "to be composed with other indices. "
                "In order to set text, please run `index.set_text()`."
            )
        return self._index_struct

    def set_text(self, text: str) -> None:
        """Set summary text for index struct.

        This allows index_struct_with_text to be used to compose indices
        with other indices.

        """
        self._index_struct.text = text

    def set_extra_info(self, extra_info: Dict[str, Any]) -> None:
        """Set extra info (metadata) for index struct.

        If this index is used as a subindex for a parent index, the metadata
        will be propagated to all nodes derived from this subindex, in the
        parent index.

        """
        self._index_struct.extra_info = extra_info

    def set_doc_id(self, doc_id: str) -> None:
        """Set doc_id for index struct.

        This is used to uniquely identify the index struct in the docstore.
        If you wish to delete the index struct, you can use this doc_id.

        """
        old_doc_id = self._index_struct.get_doc_id()
        self._index_struct.doc_id = doc_id
        # Note: we also need to delete old doc_id, and update docstore
        self._docstore.delete_document(old_doc_id)
        self._docstore.add_documents([self._index_struct], allow_update=True)

    def get_doc_id(self) -> str:
        """Get doc_id for index struct.

        If doc_id not set, raise an error.

        """
        if self._index_struct.doc_id is None:
            raise ValueError("Index must have doc_id property set.")
        return self._index_struct.doc_id

    def _get_nodes_from_document(
        self,
        document: BaseDocument,
        start_idx: int = 0,
    ) -> List[Node]:
        return get_nodes_from_document(
            document=document,
            text_splitter=self._text_splitter,
            start_idx=start_idx,
            include_extra_info=self._include_extra_info,
        )

    def _build_fallback_text_splitter(self) -> TextSplitter:
        """Build the text splitter if not specified in args."""
        return TokenTextSplitter()

    @abstractmethod
    def _build_index_from_documents(self, documents: Sequence[BaseDocument]) -> IS:
        """Build the index from documents."""

    @llm_token_counter("build_index_from_documents")
    def build_index_from_documents(self, documents: Sequence[BaseDocument]) -> IS:
        """Build the index from documents."""
        return self._build_index_from_documents(documents)

    @abstractmethod
    def _insert(self, document: BaseDocument, **insert_kwargs: Any) -> None:
        """Insert a document."""

    @llm_token_counter("insert")
    def insert(self, document: DOCUMENTS_INPUT, **insert_kwargs: Any) -> None:
        """Insert a document.

        Args:
            document (Union[BaseDocument, BaseGPTIndex]): document to insert

        """
        processed_doc = self._process_documents(
            [document], self._docstore, self._index_registry
        )[0]
        self._validate_documents([processed_doc])
        self._insert(processed_doc, **insert_kwargs)

    @abstractmethod
    def _delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document."""

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document from the index.

        All nodes in the index related to the index will be deleted.

        Args:
            doc_id (str): document id

        """
        logging.debug(f"> Deleting document: {doc_id}")
        self._delete(doc_id, **delete_kwargs)

    def update(self, document: DOCUMENTS_INPUT, **update_kwargs: Any) -> None:
        """Update a document.

        This is equivalent to deleting the document and then inserting it again.

        Args:
            document (Union[BaseDocument, BaseGPTIndex]): document to update
            insert_kwargs (Dict): kwargs to pass to insert
            delete_kwargs (Dict): kwargs to pass to delete

        """
        self.delete(document.get_doc_id(), **update_kwargs.pop("delete_kwargs", {}))
        self.insert(document, **update_kwargs.pop("insert_kwargs", {}))

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
        query_str: Union[str, QueryBundle],
        mode: str = QueryMode.DEFAULT,
        query_transform: Optional[BaseQueryTransform] = None,
        use_async: bool = False,
        **query_kwargs: Any,
    ) -> Response:
        """Answer a query.

        When `query` is called, we query the index with the given `mode` and
        `query_kwargs`. The `mode` determines the type of query to run, and
        `query_kwargs` are parameters that are specific to the query type.

        For a comprehensive documentation of available `mode` and `query_kwargs` to
        query a given index, please visit :ref:`Ref-Query`.


        """
        mode_enum = QueryMode(mode)
        if mode_enum == QueryMode.RECURSIVE:
            # TODO: deprecated, use ComposableGraph instead.
            if "query_configs" not in query_kwargs:
                raise ValueError("query_configs must be provided for recursive mode.")
            query_configs = query_kwargs["query_configs"]
            query_runner = QueryRunner(
                self._llm_predictor,
                self._prompt_helper,
                self._embed_model,
                self._docstore,
                self._index_registry,
                query_configs=query_configs,
                query_transform=query_transform,
                recursive=True,
                use_async=use_async,
            )
            return query_runner.query(query_str, self._index_struct)
        else:
            self._preprocess_query(mode_enum, query_kwargs)
            # TODO: pass in query config directly
            query_config = QueryConfig(
                index_struct_type=self._index_struct.get_type(),
                query_mode=mode_enum,
                query_kwargs=query_kwargs,
            )
            query_runner = QueryRunner(
                self._llm_predictor,
                self._prompt_helper,
                self._embed_model,
                self._docstore,
                self._index_registry,
                query_configs=[query_config],
                query_transform=query_transform,
                recursive=False,
                use_async=use_async,
            )
            return query_runner.query(query_str, self._index_struct)

    @classmethod
    @abstractmethod
    def get_query_map(cls) -> Dict[str, Type[BaseGPTIndexQuery]]:
        """Get query map."""

    @classmethod
    def load_from_dict(
        cls, result_dict: Dict[str, Any], **kwargs: Any
    ) -> "BaseGPTIndex":
        """Load index from dict."""
        if "index_struct" in result_dict:
            index_struct = cls.index_struct_cls.from_dict(result_dict["index_struct"])
            index_struct_id = index_struct.get_doc_id()
        elif "index_struct_id" in result_dict:
            index_struct_id = result_dict["index_struct_id"]
        else:
            raise ValueError("index_struct or index_struct_id must be provided.")

        type_to_struct = {cls.index_struct_cls.get_type(): cls.index_struct_cls}

        # NOTE: index_struct can have multiple types for backwards compatibility,
        # map to same class
        type_to_struct = {
            index_type: cls.index_struct_cls
            for index_type in cls.index_struct_cls.get_types()
        }

        docstore = DocumentStore.load_from_dict(
            result_dict["docstore"],
            type_to_struct=type_to_struct,
        )
        if "index_struct_id" in result_dict:
            index_struct = docstore.get_document(index_struct_id)
        return cls(index_struct=index_struct, docstore=docstore, **kwargs)

    @classmethod
    def load_from_string(cls, index_string: str, **kwargs: Any) -> "BaseGPTIndex":
        """Load index from string (in JSON-format).

        This method loads the index from a JSON string. The index data
        structure itself is preserved completely. If the index is defined over
        subindices, those subindices will also be preserved (and subindices of
        those subindices, etc.).

        NOTE: load_from_string should not be used for indices composed on top
        of other indices. Please define a `ComposableGraph` and use
        `save_to_string` and `load_from_string` on that instead.

        Args:
            index_string (str): The index string (in JSON-format).

        Returns:
            BaseGPTIndex: The loaded index.

        """
        result_dict = json.loads(index_string)
        return cls.load_from_dict(result_dict, **kwargs)

    @classmethod
    def load_from_disk(cls, save_path: str, **kwargs: Any) -> "BaseGPTIndex":
        """Load index from disk.

        This method loads the index from a JSON file stored on disk. The index data
        structure itself is preserved completely. If the index is defined over
        subindices, those subindices will also be preserved (and subindices of
        those subindices, etc.).

        NOTE: load_from_disk should not be used for indices composed on top
        of other indices. Please define a `ComposableGraph` and use
        `save_to_disk` and `load_from_disk` on that instead.

        Args:
            save_path (str): The save_path of the file.

        Returns:
            BaseGPTIndex: The loaded index.

        """
        with open(save_path, "r") as f:
            file_contents = f.read()
            return cls.load_from_string(file_contents, **kwargs)

    def save_to_dict(self, **save_kwargs: Any) -> dict:
        """Save to dict."""
        if self.docstore.contains_index_struct(
            exclude_ids=[self.index_struct.get_doc_id()]
        ):
            raise ValueError(
                "Cannot call save index if index is composed on top of "
                "other indices. Please define a `ComposableGraph` and use "
                "`save_to_string` and `load_from_string` on that instead."
            )
        out_dict: Dict[str, Any] = {
            "index_struct_id": self.index_struct.get_doc_id(),
            "docstore": self.docstore.serialize_to_dict(),
        }
        return out_dict

    def save_to_string(self, **save_kwargs: Any) -> str:
        """Save to string.

        This method stores the index into a JSON string.

        NOTE: save_to_string should not be used for indices composed on top
        of other indices. Please define a `ComposableGraph` and use
        `save_to_string` and `load_from_string` on that instead.

        Returns:
            str: The JSON string of the index.

        """
        out_dict = self.save_to_dict(**save_kwargs)
        return json.dumps(out_dict, **save_kwargs)

    def save_to_disk(self, save_path: str, **save_kwargs: Any) -> None:
        """Save to file.

        This method stores the index into a JSON file stored on disk.

        NOTE: save_to_disk should not be used for indices composed on top
        of other indices. Please define a `ComposableGraph` and use
        `save_to_disk` and `load_from_disk` on that instead.

        Args:
            save_path (str): The save_path of the file.

        """
        index_string = self.save_to_string(**save_kwargs)
        with open(save_path, "w") as f:
            f.write(index_string)
