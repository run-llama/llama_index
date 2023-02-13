"""Qdrant vector store index.

An index that is built on top of an existing Qdrant collection.

"""
from typing import Any, Dict, Optional, Sequence, Type, cast

from gpt_index.data_structs.data_structs import QdrantIndexStruct
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.indices.base import DOCUMENTS_INPUT
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.schema import QueryMode
from gpt_index.indices.query.vector_store.qdrant import GPTQdrantIndexQuery
from gpt_index.indices.vector_store.base import BaseGPTVectorStoreIndex
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from gpt_index.prompts.prompts import QuestionAnswerPrompt
from gpt_index.schema import BaseDocument
from gpt_index.utils import get_new_id


class GPTQdrantIndex(BaseGPTVectorStoreIndex[QdrantIndexStruct]):
    """GPT Qdrant Index.

    The GPTQdrantIndex is a data structure where nodes are keyed by
    embeddings, and those embeddings are stored within a Qdrant collection.
    During index construction, the document texts are chunked up,
    converted to nodes with text; they are then encoded in
    document embeddings stored within Qdrant.

    During query time, the index uses Qdrant to query for the top
    k most similar nodes, and synthesizes an answer from the
    retrieved nodes.

    Args:
        text_qa_template (Optional[QuestionAnswerPrompt]): A Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        embed_model (Optional[BaseEmbedding]): Embedding model to use for
            embedding similarity.
        client (Optional[Any]): QdrantClient instance from `qdrant-client` package
        collection_name: (Optional[str]): name of the Qdrant collection
    """

    index_struct_cls = QdrantIndexStruct

    def __init__(
        self,
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        index_struct: Optional[QdrantIndexStruct] = None,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        embed_model: Optional[BaseEmbedding] = None,
        client: Optional[Any] = None,
        collection_name: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Init params."""
        import_err_msg = (
            "`qdrant-client` package not found, please run `pip install qdrant-client`"
        )
        try:
            import qdrant_client  # noqa: F401
        except ImportError:
            raise ValueError(import_err_msg)

        if client is None:
            raise ValueError("client cannot be None.")

        if collection_name is None and index_struct is not None:
            collection_name = index_struct.collection_name
        if collection_name is None:
            raise ValueError("collection_name cannot be None.")

        self._client = cast(qdrant_client.QdrantClient, client)
        self._collection_name = collection_name
        self._collection_initialized = self._collection_exists(collection_name)

        self.text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
        super().__init__(
            documents,
            index_struct,
            text_qa_template,
            llm_predictor,
            embed_model,
            **kwargs
        )
        # NOTE: when building the vector store index, text_qa_template is not partially
        # formatted because we don't know the query ahead of time.
        self._text_splitter = self._prompt_helper.get_text_splitter_given_prompt(
            self.text_qa_template, 1
        )

    @classmethod
    def get_query_map(self) -> Dict[str, Type[BaseGPTIndexQuery]]:
        """Get query map."""
        return {
            QueryMode.DEFAULT: GPTQdrantIndexQuery,
            QueryMode.EMBEDDING: GPTQdrantIndexQuery,
        }

    def _add_document_to_index(
        self,
        index_struct: QdrantIndexStruct,
        document: BaseDocument,
        text_splitter: TokenTextSplitter,
    ) -> None:
        """Add document to index."""
        from qdrant_client.http import models as rest
        from qdrant_client.http.exceptions import UnexpectedResponse

        nodes = self._get_nodes_from_document(document, text_splitter)
        for n in nodes:
            if n.embedding is None:
                text_embedding = self._embed_model.get_text_embedding(n.get_text())
            else:
                text_embedding = n.embedding

            collection_name = index_struct.get_collection_name()

            # Create the Qdrant collection, if it does not exist yet
            if not self._collection_initialized:
                self._create_collection(
                    collection_name=collection_name,
                    vector_size=len(text_embedding),
                )
                self._collection_initialized = True

            while True:
                new_id = get_new_id(set())
                try:
                    self._client.http.points_api.get_point(
                        collection_name=collection_name, id=new_id
                    )
                except UnexpectedResponse:
                    break

            payload = {
                "doc_id": document.get_doc_id(),
                "text": n.get_text(),
                "index": n.index,
            }

            self._client.upsert(
                collection_name=collection_name,
                points=[
                    rest.PointStruct(
                        id=new_id,
                        vector=text_embedding,
                        payload=payload,
                    )
                ],
            )

    def _build_index_from_documents(
        self, documents: Sequence[BaseDocument]
    ) -> QdrantIndexStruct:
        """Build index from documents."""
        text_splitter = self._prompt_helper.get_text_splitter_given_prompt(
            self.text_qa_template, 1
        )
        index_struct = self.index_struct_cls(collection_name=self._collection_name)
        for d in documents:
            self._add_document_to_index(index_struct, d, text_splitter)
        return index_struct

    def _insert(self, document: BaseDocument, **insert_kwargs: Any) -> None:
        """Insert a document."""
        self._add_document_to_index(self.index_struct, document, self._text_splitter)

    def _delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document."""
        from qdrant_client.http import models as rest

        self._client.delete(
            collection_name=self._collection_name,
            points_selector=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="doc_id", match=rest.MatchValue(value=doc_id)
                    )
                ]
            ),
        )

    def _preprocess_query(self, mode: QueryMode, query_kwargs: Any) -> None:
        """Query mode to class."""
        super()._preprocess_query(mode, query_kwargs)
        # Pass along Qdrant client instance
        query_kwargs["client"] = self._client

    def _create_collection(self, collection_name: str, vector_size: int) -> None:
        """Create a Qdrant collection."""
        from qdrant_client.http import models as rest

        self._client.recreate_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(
                size=vector_size,
                distance=rest.Distance.COSINE,
            ),
        )

    def _collection_exists(self, collection_name: str) -> bool:
        from qdrant_client.http.exceptions import UnexpectedResponse

        try:
            response = self._client.http.collections_api.get_collection(collection_name)
            return response.result is not None
        except UnexpectedResponse:
            return False
