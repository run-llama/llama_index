from typing import Any, Sequence, Optional, cast

from gpt_index import LLMPredictor, QuestionAnswerPrompt
from gpt_index.data_structs.data_structs import QdrantIndexStruct
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.indices.base import DOCUMENTS_INPUT
from gpt_index.indices.vector_store.base import BaseGPTVectorStoreIndex
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.schema import BaseDocument


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
        import_err_msg = """
            `qdrant-client` package not found. Please run `pip install qdrant-client`
        """
        try:
            import qdrant_client  # noqa: F401
        except ImportError:
            raise ValueError(import_err_msg)

        if client is None:
            raise ValueError("client cannot be None.")

        self._client = cast(qdrant_client.QdrantClient, client)
        self._collection_name = collection_name

        super().__init__(
            documents,
            index_struct,
            text_qa_template,
            llm_predictor,
            embed_model,
            **kwargs
        )

    def _add_document_to_index(
        self,
        index_struct: QdrantIndexStruct,
        document: BaseDocument,
        text_splitter: TokenTextSplitter,
    ) -> None:
        pass

    def _delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        # TODO: should we allow putting doc_id as any string?
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=[doc_id],
        )
