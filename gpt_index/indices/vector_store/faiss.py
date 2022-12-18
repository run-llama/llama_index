"""Vector store index.

An index that that is built on top of an existing vector store.

"""

from typing import Any, Optional, Sequence, cast

import numpy as np

from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.base import DEFAULT_MODE, DOCUMENTS_INPUT, BaseGPTIndex
from gpt_index.indices.data_structs import IndexDict
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.vector_store.faiss import GPTFaissIndexQuery
from gpt_index.indices.utils import truncate_text
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.prompts.base import Prompt
from gpt_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from gpt_index.schema import BaseDocument


class GPTFaissIndex(BaseGPTIndex[IndexDict]):
    """GPT Faiss Index.

    The GPTFaissIndex is a data structure where nodes are keyed by
    embeddings, and those embeddings are stored within a Faiss index.
    During index construction, the document texts are chunked up,
    converted to nodes with text; they are then encoded in
    document embeddings stored within Faiss.

    During query time, the index uses Faiss to query for the top
    k most similar nodes, and synthesizes an answer from the
    retrieved nodes.

    Args:
        text_qa_template (Optional[Prompt]): A Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        faiss_index (faiss.Index): A Faiss Index object (required)
        embed_model (Optional[OpenAIEmbedding]): Embedding model to use for
            embedding similarity.
    """

    def __init__(
        self,
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        index_struct: Optional[IndexDict] = None,
        text_qa_template: Optional[Prompt] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        faiss_index: Optional[Any] = None,
        embed_model: Optional[OpenAIEmbedding] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        import_err_msg = """
            `faiss` package not found. For instructions on
            how to install `faiss` please visit
            https://github.com/facebookresearch/faiss/wiki/Installing-Faiss
        """
        try:
            import faiss  # noqa: F401
        except ImportError:
            raise ValueError(import_err_msg)

        self.text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
        if faiss_index is None:
            raise ValueError("faiss_index cannot be None.")
        # NOTE: cast to Any for now
        self._faiss_index = cast(Any, faiss_index)
        self._embed_model = embed_model or OpenAIEmbedding()
        super().__init__(
            documents=documents,
            index_struct=index_struct,
            llm_predictor=llm_predictor,
            **kwargs,
        )
        self._text_splitter = self._prompt_helper.get_text_splitter_given_prompt(
            self.text_qa_template, 1
        )

    def _add_document_to_index(
        self,
        index_struct: IndexDict,
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
            text_embedding_np = np.array(text_embedding)[np.newaxis, :]
            new_id = self._faiss_index.ntotal
            self._faiss_index.add(text_embedding_np)

            # add to index
            index_struct.add_text(text_chunk, document.get_doc_id(), text_id=new_id)

    def build_index_from_documents(
        self, documents: Sequence[BaseDocument]
    ) -> IndexDict:
        """Build index from documents."""
        text_splitter = self._prompt_helper.get_text_splitter_given_prompt(
            self.text_qa_template, 1
        )
        index_struct = IndexDict()
        for d in documents:
            self._add_document_to_index(index_struct, d, text_splitter)
        return index_struct

    def _mode_to_query(
        self, mode: str, *query_args: Any, **query_kwargs: Any
    ) -> BaseGPTIndexQuery:
        if mode == DEFAULT_MODE:
            if "text_qa_template" not in query_kwargs:
                query_kwargs["text_qa_template"] = self.text_qa_template
            query: GPTFaissIndexQuery = GPTFaissIndexQuery(
                self.index_struct, faiss_index=self._faiss_index, **query_kwargs
            )
        else:
            raise ValueError(f"Invalid query mode: {mode}.")
        return query
