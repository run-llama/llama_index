"""Simple vector store index."""

from typing import Any, Dict, Optional, Sequence, Type

from gpt_index.data_structs.data_structs import SimpleIndexDict
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.indices.base import DOCUMENTS_INPUT
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.schema import QueryMode
from gpt_index.indices.query.vector_store.simple import GPTSimpleVectorIndexQuery
from gpt_index.indices.vector_store.base import BaseGPTVectorStoreIndex
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.prompts.prompts import QuestionAnswerPrompt
from gpt_index.schema import BaseDocument
from gpt_index.utils import get_new_id


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

    @classmethod
    def get_query_map(self) -> Dict[str, Type[BaseGPTIndexQuery]]:
        """Get query map."""
        return {
            QueryMode.DEFAULT: GPTSimpleVectorIndexQuery,
            QueryMode.EMBEDDING: GPTSimpleVectorIndexQuery,
        }

    def _add_document_to_index(
        self,
        index_struct: SimpleIndexDict,
        document: BaseDocument,
        text_splitter: TokenTextSplitter,
    ) -> None:
        """Add document to index."""
        nodes = self._get_nodes_from_document(document, text_splitter)
        for n in nodes:
            # add to in-memory dict
            # NOTE: embeddings won't be stored in Node but rather in underlying
            # Faiss store
            if n.embedding is None:
                text_embedding = self._embed_model.get_text_embedding(n.get_text())
            else:
                text_embedding = n.embedding
            new_id = get_new_id(set(index_struct.nodes_dict.keys()))

            # add to index
            index_struct.add_node(n, text_id=new_id)
            # TODO: deprecate
            index_struct.add_to_embedding_dict(new_id, text_embedding)

    def _delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document."""
        text_ids_to_delete = set()
        int_ids_to_delete = set()
        for text_id, int_id in self.index_struct.id_map.items():
            node = self.index_struct.nodes_dict[int_id]
            if node.ref_doc_id != doc_id:
                continue
            text_ids_to_delete.add(text_id)
            int_ids_to_delete.add(int_id)

        for int_id, text_id in zip(int_ids_to_delete, text_ids_to_delete):
            del self.index_struct.nodes_dict[int_id]
            del self.index_struct.id_map[text_id]
            del self.index_struct.embedding_dict[text_id]
