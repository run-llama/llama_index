"""Local RAG CLI Pack."""

from llama_index.cli.rag import RagCLI
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.query_pipeline.query import QueryPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.utils import get_cache_dir
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.query_pipeline import InputComponent
from llama_index.core.llama_pack.base import BaseLlamaPack
from typing import Optional, Dict, Any
from pathlib import Path
import chromadb


def default_ragcli_persist_dir() -> str:
    """Get default RAG CLI persist dir."""
    return str(Path(get_cache_dir()) / "rag_cli_local")


def init_local_rag_cli(
    persist_dir: Optional[str] = None,
    verbose: bool = False,
    llm_model_name: str = "mistral",
    embed_model_name: str = "BAAI/bge-m3",
) -> RagCLI:
    """Init local RAG CLI."""
    docstore = SimpleDocumentStore()
    persist_dir = persist_dir or default_ragcli_persist_dir()
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.create_collection("default", get_or_create=True)
    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection, persist_dir=persist_dir
    )
    print("> Chroma collection initialized")
    llm = Ollama(model=llm_model_name, request_timeout=30.0)
    print("> LLM initialized")
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name, pooling="mean")
    print("> Embedding model initialized")

    ingestion_pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(), embed_model],
        vector_store=vector_store,
        docstore=docstore,
        cache=IngestionCache(),
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
    retriever = VectorStoreIndex.from_vector_store(
        ingestion_pipeline.vector_store
    ).as_retriever(similarity_top_k=8)
    response_synthesizer = CompactAndRefine(streaming=True, verbose=True)
    # define query pipeline
    query_pipeline = QueryPipeline(verbose=verbose)
    query_pipeline.add_modules(
        {
            "input": InputComponent(),
            "retriever": retriever,
            "summarizer": response_synthesizer,
        }
    )
    query_pipeline.add_link("input", "retriever")
    query_pipeline.add_link("retriever", "summarizer", dest_key="nodes")
    query_pipeline.add_link("input", "summarizer", dest_key="query_str")

    return RagCLI(
        ingestion_pipeline=ingestion_pipeline,
        llm=llm,  # optional
        persist_dir=persist_dir,
        query_pipeline=query_pipeline,
        verbose=False,
    )


class LocalRAGCLIPack(BaseLlamaPack):
    """Local RAG CLI Pack."""

    def __init__(
        self,
        verbose: bool = False,
        persist_dir: Optional[str] = None,
        llm_model_name: str = "mistral",
        embed_model_name: str = "BAAI/bge-m3",
    ) -> None:
        """Init params."""
        self.verbose = verbose
        self.persist_dir = persist_dir or default_ragcli_persist_dir()
        self.llm_model_name = llm_model_name
        self.embed_model_name = embed_model_name
        self.rag_cli = init_local_rag_cli(
            persist_dir=self.persist_dir,
            verbose=self.verbose,
            llm_model_name=self.llm_model_name,
            embed_model_name=self.embed_model_name,
        )

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {"rag_cli": self.rag_cli}

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return self.rag_cli.cli(*args, **kwargs)


if __name__ == "__main__":
    rag_cli_instance = init_local_rag_cli()
    rag_cli_instance.cli()
