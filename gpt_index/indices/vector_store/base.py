"""Base vector store index.

An index that that is built on top of an existing vector store.

"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Type

from gpt_index.async_utils import run_async_tasks
from gpt_index.data_structs.data_structs import IndexDict, Node
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.indices.base import DOCUMENTS_INPUT, BaseGPTIndex
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.schema import QueryMode
from gpt_index.indices.query.vector_store.base import GPTVectorStoreIndexQuery
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.text_splitter import TextSplitter
from gpt_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from gpt_index.prompts.prompts import QuestionAnswerPrompt
from gpt_index.schema import BaseDocument
from gpt_index.utils import get_new_id
from gpt_index.vector_stores.simple import SimpleVectorStore
from gpt_index.vector_stores.types import NodeEmbeddingResult, VectorStore

VECTOR_STORE_CONFIG_DICT_KEY = "vector_store"


class GPTVectorStoreIndex(BaseGPTIndex[IndexDict]):
    """Base GPT Vector Store Index.

    Args:
        text_qa_template (Optional[QuestionAnswerPrompt]): A Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        embed_model (Optional[BaseEmbedding]): Embedding model to use for
            embedding similarity.
        vector_store (Optional[VectorStore]): Vector store to use for
            embedding similarity. See :ref:`Ref-Indices-VectorStore-Stores`
            for more details.
        use_async (bool): Whether to use asynchronous calls. Defaults to False.

    """

    index_struct_cls = IndexDict

    def __init__(
        self,
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        index_struct: Optional[IndexDict] = None,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        embed_model: Optional[BaseEmbedding] = None,
        vector_store: Optional[VectorStore] = None,
        text_splitter: Optional[TextSplitter] = None,
        use_async: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._vector_store = vector_store or SimpleVectorStore()

        self.text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
        self._use_async = use_async
        super().__init__(
            documents=documents,
            index_struct=index_struct,
            llm_predictor=llm_predictor,
            embed_model=embed_model,
            text_splitter=text_splitter,
            **kwargs,
        )

    @classmethod
    def get_query_map(self) -> Dict[str, Type[BaseGPTIndexQuery]]:
        """Get query map."""
        return {
            QueryMode.DEFAULT: GPTVectorStoreIndexQuery,
            QueryMode.EMBEDDING: GPTVectorStoreIndexQuery,
        }

    def _get_node_embedding_results(
        self, nodes: List[Node], existing_node_ids: Set, doc_id: str
    ) -> List[NodeEmbeddingResult]:
        """Get tuples of id, node, and embedding.

        Allows us to store these nodes in a vector store.
        Embeddings are called in batches.

        """
        id_to_node_map: Dict[str, Node] = {}
        id_to_embed_map: Dict[str, List[float]] = {}

        for n in nodes:
            new_id = get_new_id(existing_node_ids.union(id_to_node_map.keys()))
            if n.embedding is None:
                self._embed_model.queue_text_for_embeddding(new_id, n.get_text())
            else:
                id_to_embed_map[new_id] = n.embedding

            id_to_node_map[new_id] = n

        # call embedding model to get embeddings
        result_ids, result_embeddings = self._embed_model.get_queued_text_embeddings()
        for new_id, text_embedding in zip(result_ids, result_embeddings):
            id_to_embed_map[new_id] = text_embedding

        result_tups = []
        for id, embed in id_to_embed_map.items():
            result_tups.append(
                NodeEmbeddingResult(id, id_to_node_map[id], embed, doc_id=doc_id)
            )
        return result_tups

    async def _aget_node_embedding_results(
        self, nodes: List[Node], existing_node_ids: Set, doc_id: str
    ) -> List[NodeEmbeddingResult]:
        """Asynchronously get tuples of id, node, and embedding.

        Allows us to store these nodes in a vector store.
        Embeddings are called in batches.

        """
        id_to_node_map: Dict[str, Node] = {}
        id_to_embed_map: Dict[str, List[float]] = {}

        text_queue: List[Tuple[str, str]] = []
        for n in nodes:
            new_id = get_new_id(existing_node_ids.union(id_to_node_map.keys()))
            if n.embedding is None:
                text_queue.append((new_id, n.get_text()))
            else:
                id_to_embed_map[new_id] = n.embedding

            id_to_node_map[new_id] = n

        # call embedding model to get embeddings
        (
            result_ids,
            result_embeddings,
        ) = await self._embed_model.aget_queued_text_embeddings(text_queue)
        for new_id, text_embedding in zip(result_ids, result_embeddings):
            id_to_embed_map[new_id] = text_embedding

        result_tups = []
        for id, embed in id_to_embed_map.items():
            result_tups.append(
                NodeEmbeddingResult(id, id_to_node_map[id], embed, doc_id=doc_id)
            )
        return result_tups

    def _build_fallback_text_splitter(self) -> TextSplitter:
        # if not specified, use "smart" text splitter to ensure chunks fit in prompt
        return self._prompt_helper.get_text_splitter_given_prompt(
            self.text_qa_template, 1
        )

    async def _async_add_document_to_index(
        self,
        index_struct: IndexDict,
        document: BaseDocument,
    ) -> None:
        """Asynchronously add document to index."""
        nodes = self._get_nodes_from_document(document)
        embedding_results = await self._aget_node_embedding_results(
            nodes, set(), document.get_doc_id()
        )

        new_ids = self._vector_store.add(embedding_results)

        # if the vector store doesn't store text, we need to add the nodes to the
        # index struct
        if not self._vector_store.stores_text:
            for result, new_id in zip(embedding_results, new_ids):
                index_struct.add_node(result.node, text_id=new_id)

    def _add_document_to_index(
        self,
        index_struct: IndexDict,
        document: BaseDocument,
    ) -> None:
        """Add document to index."""
        nodes = self._get_nodes_from_document(document)
        embedding_results = self._get_node_embedding_results(
            nodes, set(), document.get_doc_id()
        )

        new_ids = self._vector_store.add(embedding_results)

        # if the vector store doesn't store text, we need to add the nodes to the
        # index struct
        if not self._vector_store.stores_text:
            for result, new_id in zip(embedding_results, new_ids):
                index_struct.add_node(result.node, text_id=new_id)

    def _build_index_from_documents(
        self, documents: Sequence[BaseDocument]
    ) -> IndexDict:
        """Build index from documents."""
        index_struct = self.index_struct_cls()
        if self._use_async:
            tasks = [
                self._async_add_document_to_index(index_struct, d) for d in documents
            ]
            run_async_tasks(tasks)
        else:
            for d in documents:
                self._add_document_to_index(index_struct, d)
        return index_struct

    def _insert(self, document: BaseDocument, **insert_kwargs: Any) -> None:
        """Insert a document."""
        self._add_document_to_index(self._index_struct, document)

    def _delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document."""
        self._index_struct.delete(doc_id)
        self._vector_store.delete(doc_id)

    @classmethod
    def load_from_dict(
        cls, result_dict: Dict[str, Any], **kwargs: Any
    ) -> "BaseGPTIndex":
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
        config_dict = {}
        if "vector_store" in result_dict:
            config_dict = result_dict[VECTOR_STORE_CONFIG_DICT_KEY]
        return super().load_from_dict(result_dict, **config_dict, **kwargs)

    def save_to_dict(self, **save_kwargs: Any) -> dict:
        """Save to string.

        This method stores the index into a JSON string.

        NOTE: save_to_string should not be used for indices composed on top
        of other indices. Please define a `ComposableGraph` and use
        `save_to_string` and `load_from_string` on that instead.

        Returns:
            dict: The JSON dict of the index.

        """
        out_dict = super().save_to_dict()
        out_dict[VECTOR_STORE_CONFIG_DICT_KEY] = self._vector_store.config_dict
        return out_dict

    def _preprocess_query(self, mode: QueryMode, query_kwargs: Any) -> None:
        super()._preprocess_query(mode, query_kwargs)
        # NOTE: Pass along vector store instance to query objects
        # TODO: refactor this to be more explicit
        query_kwargs["vector_store"] = self._vector_store
