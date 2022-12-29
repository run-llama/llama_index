"""Simple vector store index."""

from typing import Any, Optional, Sequence

from gpt_index.data_structs.data_structs import SimpleIndexDict
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.indices.base import DOCUMENTS_INPUT
from gpt_index.indices.utils import truncate_text
from gpt_index.indices.vector_store.base import BaseGPTVectorStoreIndex
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.prompts.prompts import QuestionAnswerPrompt
from gpt_index.schema import BaseDocument


class GPTSimpleVectorIndex(BaseGPTVectorStoreIndex[SimpleIndexDict]):
    """GPT Simple Vector Index.

    The GPTSimpleVectorIndex is a data structure where nodes are keyed by
    embeddings, and those embeddings are stored within a simple dictionary.
    During index construction, the document texts are chunked up,
    converted to nodes with text; they are then encoded in
    document embeddings stored within the dict.

    During query time, the index uses the dict to query for the top
    k most similar nodes, and synthesizes an answer from the
    retrieved nodes.

    Args:
        text_qa_template (Optional[QuestionAnswerPrompt]): A Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        embed_model (Optional[BaseEmbedding]): Embedding model to use for
            embedding similarity.
    """

    index_struct_cls = SimpleIndexDict

    def __init__(
        self,
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        index_struct: Optional[SimpleIndexDict] = None,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        embed_model: Optional[BaseEmbedding] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(
            documents=documents,
            index_struct=index_struct,
            text_qa_template=text_qa_template,
            llm_predictor=llm_predictor,
            embed_model=embed_model,
            **kwargs,
        )

    def _add_document_to_index(
        self,
        index_struct: SimpleIndexDict,
        document: BaseDocument,
        text_splitter: TokenTextSplitter,
    ) -> None:
        """Add document to index."""
        text_chunks = text_splitter.split_text(document.get_text())
        for _, text_chunk in enumerate(text_chunks):
            fmt_text_chunk = truncate_text(text_chunk, 50)
            print(f"> Adding chunk: {fmt_text_chunk}")
            # add to FAISS
            # NOTE: embeddings won't be stored in Node but rather in underlying
            # Faiss store
            text_embedding = self._embed_model.get_text_embedding(text_chunk)
            new_id = str(len(index_struct.nodes_dict))

            # add to index
            index_struct.add_text(text_chunk, document.get_doc_id(), text_id=new_id)
            index_struct.add_embedding(new_id, text_embedding)
