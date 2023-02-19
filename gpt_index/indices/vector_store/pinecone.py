"""Pinecone Vector store index.

An index that that is built on top of an existing vector store.

"""

from typing import Any, Dict, Optional, Sequence, Type, cast

from gpt_index.data_structs.data_structs import PineconeIndexStruct
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.indices.base import DOCUMENTS_INPUT
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.schema import QueryMode
from gpt_index.indices.query.vector_store.pinecone import GPTPineconeIndexQuery
from gpt_index.indices.vector_store.base import BaseGPTVectorStoreIndex
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from gpt_index.prompts.prompts import QuestionAnswerPrompt
from gpt_index.schema import BaseDocument
from gpt_index.utils import get_new_id


class GPTPineconeIndex(BaseGPTVectorStoreIndex[PineconeIndexStruct]):
    """GPT Pinecone Index.

    The GPTPineconeIndex is a data structure where nodes are keyed by
    embeddings, and those embeddings are stored within a Pinecone index.
    During index construction, the document texts are chunked up,
    converted to nodes with text; they are then encoded in
    document embeddings stored within Pinecone.

    During query time, the index uses Pinecone to query for the top
    k most similar nodes, and synthesizes an answer from the
    retrieved nodes.

    Args:
        text_qa_template (Optional[QuestionAnswerPrompt]): A Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        embed_model (Optional[BaseEmbedding]): Embedding model to use for
            embedding similarity.
        chunk_size_limit (int): Maximum number of tokens per chunk. NOTE:
            in Pinecone the default is 2048 due to metadata size restrictions.
    """

    index_struct_cls = PineconeIndexStruct

    def __init__(
        self,
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        index_struct: Optional[PineconeIndexStruct] = None,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        embed_model: Optional[BaseEmbedding] = None,
        pinecone_index: Optional[Any] = None,
        chunk_size_limit: int = 2048,
        pinecone_kwargs: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        import_err_msg = (
            "`pinecone` package not found, please run `pip install pinecone-client`"
        )
        try:
            import pinecone  # noqa: F401
        except ImportError:
            raise ValueError(import_err_msg)
        self._pinecone_index = cast(pinecone.Index, pinecone_index)

        self._pinecone_kwargs = pinecone_kwargs or {}

        self.text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
        super().__init__(
            documents=documents,
            index_struct=index_struct,
            llm_predictor=llm_predictor,
            embed_model=embed_model,
            chunk_size_limit=chunk_size_limit,
            **kwargs,
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
            QueryMode.DEFAULT: GPTPineconeIndexQuery,
            QueryMode.EMBEDDING: GPTPineconeIndexQuery,
        }

    def _add_document_to_index(
        self,
        index_struct: PineconeIndexStruct,
        document: BaseDocument,
        text_splitter: TokenTextSplitter,
    ) -> None:
        """Add document to index."""
        nodes = self._get_nodes_from_document(document, text_splitter)

        id_node_embed_tups = self._get_node_embedding_tups(nodes, set())
        for new_id, node, text_embedding in id_node_embed_tups:
            # assign a new_id if current_id conflicts with existing ids
            while True:
                result = self._pinecone_index.fetch([new_id], **self._pinecone_kwargs)
                if len(result["vectors"]) == 0:
                    break
                new_id = get_new_id(set())
            metadata = {
                "text": node.get_text(),
                "doc_id": document.get_doc_id(),
            }
            self._pinecone_index.upsert(
                [(new_id, text_embedding, metadata)], **self._pinecone_kwargs
            )

    def _delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document."""
        # delete by filtering on the doc_id metadata
        self._pinecone_index.delete(
            filter={"doc_id": {"$eq": doc_id}}, **self._pinecone_kwargs
        )

    def _preprocess_query(self, mode: QueryMode, query_kwargs: Any) -> None:
        """Query mode to class."""
        super()._preprocess_query(mode, query_kwargs)
        # pass along pinecone client and info
        query_kwargs["pinecone_index"] = self._pinecone_index
