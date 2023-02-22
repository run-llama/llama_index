"""Chroma vector store index."""

from typing import Any, Dict, Optional, Sequence, Type, cast

from gpt_index.data_structs.data_structs import ChromaIndexStruct
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.indices.base import DOCUMENTS_INPUT
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.schema import QueryMode
from gpt_index.indices.query.vector_store.chroma import GPTChromaIndexQuery
from gpt_index.indices.vector_store.base import BaseGPTVectorStoreIndex
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.text_splitter import TextSplitter
from gpt_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from gpt_index.prompts.prompts import QuestionAnswerPrompt
from gpt_index.schema import BaseDocument


class GPTChromaIndex(BaseGPTVectorStoreIndex[ChromaIndexStruct]):
    """GPT Chroma Index.

    The GPTChromaIndex is a data structure where nodes are keyed by
    embeddings, and those embeddings are stored within a Chroma collection.
    During index construction, the document texts are chunked up,
    converted to nodes with text; they are then encoded in
    document embeddings stored within Chroma.

    During query time, the index uses Chroma to query for the top
    k most similar nodes, and synthesizes an answer from the
    retrieved nodes.

    Args:
        text_qa_template (Optional[QuestionAnswerPrompt]): A Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        embed_model (Optional[BaseEmbedding]): Embedding model to use for
            embedding similarity.
        chroma_collection (Optional[Any]): Collection instance from `chromadb` package

    """

    index_struct_cls = ChromaIndexStruct

    def __init__(
        self,
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        index_struct: Optional[ChromaIndexStruct] = None,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        embed_model: Optional[BaseEmbedding] = None,
        text_splitter: Optional[TextSplitter] = None,
        chroma_collection: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        import_err_msg = (
            "`chromadb` package not found, please run `pip install chromadb`"
        )
        try:
            import chromadb  # noqa: F401
        except ImportError:
            raise ValueError(import_err_msg)
        from chromadb.api.models.Collection import Collection

        self._collection = cast(Collection, chroma_collection)

        self.text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
        super().__init__(
            documents=documents,
            index_struct=index_struct,
            text_qa_template=text_qa_template,
            llm_predictor=llm_predictor,
            embed_model=embed_model,
            text_splitter=text_splitter,
            **kwargs,
        )

    @classmethod
    def get_query_map(cls) -> Dict[str, Type[BaseGPTIndexQuery]]:
        """Get query map."""
        return {
            QueryMode.DEFAULT: GPTChromaIndexQuery,
            QueryMode.EMBEDDING: GPTChromaIndexQuery,
        }

    def _add_document_to_index(
        self,
        index_struct: ChromaIndexStruct,
        document: BaseDocument,
    ) -> None:
        """Add document to index."""
        if not self._collection:
            raise ValueError("Collection not initialized")
        nodes = self._get_nodes_from_document(document)

        esxisting_ids = set(self._collection.get()["ids"])

        id_node_embed_tups = self._get_node_embedding_tups(nodes, esxisting_ids)

        embeddings = []
        metadatas = []
        ids = []
        documents = []

        for node_id, node, embed in id_node_embed_tups:
            embeddings.append(embed)
            metadatas.append({"document_id": document.get_doc_id()})
            ids.append(node_id)
            documents.append(node.get_text())

        self._collection.add(
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
            documents=documents,
        )

    def _build_index_from_documents(
        self, documents: Sequence[BaseDocument]
    ) -> ChromaIndexStruct:
        """Build index from documents."""
        index_struct = self.index_struct_cls()

        for d in documents:
            self._add_document_to_index(index_struct, d)

        return index_struct

    def _delete(self, doc_id: str, **kwargs: Any) -> None:
        """Delete a document."""
        self._collection.delete(where={"document_id": doc_id})

    def _preprocess_query(self, mode: QueryMode, query_kwargs: Any) -> None:
        """Query mode to class."""
        super()._preprocess_query(mode, query_kwargs)
        # pass along chroma collection
        query_kwargs["chroma_collection"] = self._collection
