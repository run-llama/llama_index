"""Base index classes."""
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Sequence, Type, TypeVar, Union

from gpt_index.constants import DOCSTORE_KEY, INDEX_STRUCT_KEY
from gpt_index.data_structs.data_structs_v2 import V2IndexStruct
from gpt_index.data_structs.node_v2 import IndexNode, Node
from gpt_index.docstore import DocumentStore
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.query_runner import QueryRunner
from gpt_index.indices.query.query_transform.base import BaseQueryTransform
from gpt_index.indices.query.schema import QueryBundle, QueryConfig, QueryMode
from gpt_index.indices.service_context import ServiceContext
from gpt_index.readers.schema.base import Document
from gpt_index.response.schema import RESPONSE_TYPE, Response
from gpt_index.token_counter.token_counter import llm_token_counter

IS = TypeVar("IS", bound=V2IndexStruct)

logger = logging.getLogger(__name__)


# map from mode to query class
QueryMap = Dict[str, Type[BaseGPTIndexQuery]]


class BaseGPTIndex(Generic[IS], ABC):

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
        nodes: Optional[Sequence[Node]] = None,
        index_struct: Optional[IS] = None,
        docstore: Optional[DocumentStore] = None,
        service_context: Optional[ServiceContext] = None,
    ) -> None:
        """Initialize with parameters."""
        if index_struct is None and nodes is None:
            raise ValueError("One of documents or index_struct must be provided.")
        if index_struct is not None and nodes is not None:
            raise ValueError("Only one of documents or index_struct can be provided.")

        self._service_context = service_context or ServiceContext.from_defaults()
        self._docstore = docstore or DocumentStore()

        if index_struct is None:
            assert nodes is not None
            self._docstore.add_documents(nodes)
            index_struct = self.build_index_from_nodes(nodes)
            # if not isinstance(index_struct, self.index_struct_cls):
            #     raise ValueError(
            #         f"index_struct must be of type {self.index_struct_cls} "
            #         f"but got {type(index_struct)}"
            #     )
        self._index_struct = index_struct

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[Document],
        docstore: Optional[DocumentStore] = None,
        service_context: Optional[ServiceContext] = None,
        **kwargs: Any,
    ) -> "BaseGPTIndex":
        """Create index from documents."""
        service_context = service_context or ServiceContext.from_defaults()
        docstore = docstore or DocumentStore()

        for doc in documents:
            docstore.set_document_hash(doc.get_doc_id(), doc.get_doc_hash())

        nodes = service_context.node_parser.get_nodes_from_documents(documents)

        return cls(
            nodes=nodes,
            docstore=docstore,
            service_context=service_context,
            **kwargs,
        )

    @classmethod
    def from_indices(
        cls,
        children_indices: Sequence["BaseGPTIndex"],
        index_summaries: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> "ComposableGraph":  # type: ignore
        """Create composable graph using this index class as the root.

        NOTE: this is mostly syntactic sugar,
        roughly equivalent to directly calling `ComposableGraph.from_indices`.
        """
        # NOTE: lazy import
        from gpt_index.indices.composability.graph import ComposableGraph

        if index_summaries is None:
            # TODO: automatically set summaries
            index_summaries = ["stub" for _ in children_indices]

        if len(children_indices) != len(index_summaries):
            raise ValueError("indices and index_summaries must have same length!")

        index_nodes = []
        for index, summary in zip(children_indices, index_summaries):
            assert isinstance(index.index_struct, V2IndexStruct)
            index_node = IndexNode(
                text=summary,
                index_id=index.index_struct.index_id,
            )
            index_nodes.append(index_node)

        root_index = cls(
            nodes=index_nodes,
            **kwargs,
        )
        all_indices: List["BaseGPTIndex"] = children_indices + [root_index]  # type: ignore
        return ComposableGraph.from_indices(
            all_index_structs={
                index.index_struct.index_id: index.index_struct for index in all_indices
            },
            root_id=root_index.index_struct.index_id,
            docstores=[index.docstore for index in all_indices],
        )

    @property
    def index_struct(self) -> IS:
        """Get the index struct."""
        return self._index_struct

    @property
    def docstore(self) -> DocumentStore:
        """Get the docstore corresponding to the index."""
        return self._docstore

    @property
    def service_context(self) -> ServiceContext:
        return self._service_context

    @abstractmethod
    def _build_index_from_nodes(self, nodes: Sequence[Node]) -> IS:
        """Build the index from nodes."""

    @llm_token_counter("build_index_from_nodes")
    def build_index_from_nodes(self, nodes: Sequence[Node]) -> IS:
        """Build the index from nodes."""
        return self._build_index_from_nodes(nodes)

    @abstractmethod
    def _insert(self, nodes: Sequence[Node], **insert_kwargs: Any) -> None:
        """Insert nodes."""

    @llm_token_counter("insert")
    def insert(self, document: Document, **insert_kwargs: Any) -> None:
        """Insert a document.

        Args:
            document (Union[BaseDocument, BaseGPTIndex]): document to insert

        """
        nodes = self.service_context.node_parser.get_nodes_from_documents([document])
        self.docstore.add_documents(nodes)
        self._insert(nodes, **insert_kwargs)

    @abstractmethod
    def _delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document."""

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document from the index.

        All nodes in the index related to the index will be deleted.

        Args:
            doc_id (str): document id

        """
        logger.debug(f"> Deleting document: {doc_id}")
        self._delete(doc_id, **delete_kwargs)

    def update(self, document: Document, **update_kwargs: Any) -> None:
        """Update a document.

        This is equivalent to deleting the document and then inserting it again.

        Args:
            document (Union[BaseDocument, BaseGPTIndex]): document to update
            insert_kwargs (Dict): kwargs to pass to insert
            delete_kwargs (Dict): kwargs to pass to delete

        """
        self.delete(document.get_doc_id(), **update_kwargs.pop("delete_kwargs", {}))
        self.insert(document, **update_kwargs.pop("insert_kwargs", {}))

    def refresh(
        self, documents: Sequence[Document], **update_kwargs: Any
    ) -> List[bool]:
        """Refresh an index with documents that have changed.

        This allows users to save LLM and Embedding model calls, while only
        updating documents that have any changes in text or extra_info. It
        will also insert any documents that previously were not stored.
        """
        refreshed_documents = [False] * len(documents)
        for i, document in enumerate(documents):
            existing_doc_hash = self._docstore.get_document_hash(document.get_doc_id())
            if existing_doc_hash != document.get_doc_hash():
                self.update(document, **update_kwargs)
                refreshed_documents[i] = True
            elif existing_doc_hash is None:
                self.insert(document, **update_kwargs.pop("insert_kwargs", {}))
                refreshed_documents[i] = True

        return refreshed_documents

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
    ) -> RESPONSE_TYPE:
        """Answer a query.

        When `query` is called, we query the index with the given `mode` and
        `query_kwargs`. The `mode` determines the type of query to run, and
        `query_kwargs` are parameters that are specific to the query type.

        For a comprehensive documentation of available `mode` and `query_kwargs` to
        query a given index, please visit :ref:`Ref-Query`.


        """
        mode_enum = QueryMode(mode)
        self._preprocess_query(mode_enum, query_kwargs)
        # TODO: pass in query config directly
        query_config = QueryConfig(
            index_struct_type=self._index_struct.get_type(),
            query_mode=mode_enum,
            query_kwargs=query_kwargs,
        )
        query_runner = QueryRunner(
            index_struct=self._index_struct,
            service_context=self._service_context,
            docstore=self._docstore,
            query_configs=[query_config],
            query_transform=query_transform,
            recursive=False,
            use_async=use_async,
        )
        return query_runner.query(query_str)

    async def aquery(
        self,
        query_str: Union[str, QueryBundle],
        mode: str = QueryMode.DEFAULT,
        query_transform: Optional[BaseQueryTransform] = None,
        **query_kwargs: Any,
    ) -> RESPONSE_TYPE:
        """Asynchronously answer a query.

        When `query` is called, we query the index with the given `mode` and
        `query_kwargs`. The `mode` determines the type of query to run, and
        `query_kwargs` are parameters that are specific to the query type.

        For a comprehensive documentation of available `mode` and `query_kwargs` to
        query a given index, please visit :ref:`Ref-Query`.


        """
        # TODO: currently we don't have async versions of all
        # underlying functions. Setting use_async=True
        # will cause async nesting errors because we assume
        # it's called in a synchronous setting.
        use_async = False

        mode_enum = QueryMode(mode)
        self._preprocess_query(mode_enum, query_kwargs)
        # TODO: pass in query config directly
        query_config = QueryConfig(
            index_struct_type=self._index_struct.get_type(),
            query_mode=mode_enum,
            query_kwargs=query_kwargs,
        )
        query_runner = QueryRunner(
            self._index_struct,
            self._service_context,
            self._docstore,
            query_configs=[query_config],
            query_transform=query_transform,
            recursive=False,
            use_async=use_async,
        )
        return await query_runner.aquery(query_str)

    @classmethod
    @abstractmethod
    def get_query_map(cls) -> QueryMap:
        """Get query map."""

    @classmethod
    def load_from_dict(
        cls, result_dict: Dict[str, Any], **kwargs: Any
    ) -> "BaseGPTIndex":
        """Load index from dict."""
        # NOTE: lazy load registry
        from gpt_index.indices.registry import load_index_struct_from_dict

        index_struct = load_index_struct_from_dict(result_dict[INDEX_STRUCT_KEY])
        assert isinstance(index_struct, cls.index_struct_cls)
        docstore = DocumentStore.load_from_dict(result_dict[DOCSTORE_KEY])
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
        out_dict: Dict[str, Any] = {
            INDEX_STRUCT_KEY: self.index_struct.to_dict(),
            DOCSTORE_KEY: self.docstore.serialize_to_dict(),
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

    def save_to_disk(
        self, save_path: str, encoding: str = "ascii", **save_kwargs: Any
    ) -> None:
        """Save to file.

        This method stores the index into a JSON file stored on disk.

        NOTE: save_to_disk should not be used for indices composed on top
        of other indices. Please define a `ComposableGraph` and use
        `save_to_disk` and `load_from_disk` on that instead.

        Args:
            save_path (str): The save_path of the file.
            encoding (str): The encoding of the file.

        """
        index_string = self.save_to_string(**save_kwargs)
        with open(save_path, "wt", encoding=encoding) as f:
            f.write(index_string)
