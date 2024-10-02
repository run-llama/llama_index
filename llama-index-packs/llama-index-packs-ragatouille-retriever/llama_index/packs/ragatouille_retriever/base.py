"""RAGatouille Retriever Pack."""

from typing import Any, Dict, List, Optional

from llama_index.core import Settings
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.llms.llm import LLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import Document, NodeWithScore, TextNode
from llama_index.llms.openai import OpenAI


class CustomRetriever(BaseRetriever):
    """Custom retriever."""

    def __init__(
        self, rag_obj: Any, index_name: str, top_k: int = 10, **kwargs: Any
    ) -> None:
        """Init params."""
        try:
            import ragatouille  # noqa
        except ImportError:
            raise ValueError(
                "RAGatouille is not installed. Please install it with `pip install ragatouille`."
            )
        self.rag_obj = rag_obj
        self.index_name = index_name
        self.top_k = top_k

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        results = self.rag_obj.search(
            query_bundle.query_str, index_name=self.index_name, k=self.top_k
        )
        result_nodes = []
        for result in results:
            result_nodes.append(
                NodeWithScore(
                    node=TextNode(text=result["content"]), score=result["score"]
                )
            )
        return result_nodes


class RAGatouilleRetrieverPack(BaseLlamaPack):
    """RAGatouille Retriever pack."""

    def __init__(
        self,
        documents: List[Document],
        model_name: str = "colbert-ir/colbertv2.0",
        index_name: str = "my_index",
        llm: Optional[LLM] = None,
        index_path: Optional[str] = None,
        top_k: int = 10,
    ) -> None:
        """Init params."""
        self._model_name = model_name
        try:
            import ragatouille  # noqa
            from ragatouille import RAGPretrainedModel
        except ImportError:
            raise ValueError(
                "RAGatouille is not installed. Please install it with `pip install ragatouille`."
            )

        doc_txts = [doc.get_content() for doc in documents]
        doc_ids = [doc.doc_id for doc in documents]
        doc_metadatas = [doc.metadata for doc in documents]

        # index the documents
        if index_path is None:
            RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
            index_path = RAG.index(
                index_name=index_name,
                collection=doc_txts,
                document_ids=doc_ids,
                document_metadatas=doc_metadatas,
            )
        else:
            RAG = RAGPretrainedModel.from_index(index_path)

        self.index_path = index_path

        self.custom_retriever = CustomRetriever(RAG, index_name=index_name, top_k=top_k)

        self.RAG = RAG
        self.documents = documents

        self.llm = llm or OpenAI(model="gpt-3.5-turbo")
        Settings.llm = self.llm
        self.query_engine = RetrieverQueryEngine.from_args(self.custom_retriever)

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents."""
        doc_txts = [doc.get_content() for doc in documents]
        doc_ids = [doc.doc_id for doc in documents]
        doc_metadatas = [doc.metadata for doc in documents]

        self.RAG.add_to_index(
            new_collection=doc_txts,
            new_document_ids=doc_ids,
            new_document_metadatas=doc_metadatas,
        )

    def delete_documents(self, documents: List[Document]) -> None:
        """Delete documents."""
        doc_ids = [doc.doc_id for doc in documents]

        self.RAG.delete_from_index(document_ids=doc_ids)

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "RAG": self.RAG,
            "documents": self.documents,
            "retriever": self.custom_retriever,
            "llm": self.llm,
            "query_engine": self.query_engine,
            "index_path": self.index_path,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return self.query_engine.query(*args, **kwargs)
