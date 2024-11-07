from typing import Any, Dict, List

from llama_index.core import Settings, VectorStoreIndex, get_response_synthesizer
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import Document
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.vector_stores.types import ExactMatchFilter, MetadataFilters
from llama_index.llms.openai import OpenAI


class MultiTenancyRAGPack(BaseLlamaPack):
    def __init__(self) -> None:
        llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
        self.llm = llm
        Settings.llm = self.llm
        self.index = VectorStoreIndex.from_documents(documents=[])

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {"llm": self.llm, "index": self.index}

    def add(self, documents: List[Document], user: Any) -> None:
        """Insert Documents of a user into index."""
        # Add metadata to documents
        for document in documents:
            document.metadata["user"] = user
        # Create Nodes using IngestionPipeline
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=512, chunk_overlap=20),
            ]
        )
        nodes = pipeline.run(documents=documents, num_workers=4)
        # Insert nodes into the index
        self.index.insert_nodes(nodes)

    def run(self, query_str: str, user: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        # Define retriever to filter out nodes for user and query
        retriever = VectorIndexRetriever(
            index=self.index,
            filters=MetadataFilters(
                filters=[
                    ExactMatchFilter(
                        key="user",
                        value=user,
                    )
                ]
            ),
            **kwargs,
        )
        # Define response synthesizer
        response_synthesizer = get_response_synthesizer(response_mode="compact")
        # Define Query Engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever, response_synthesizer=response_synthesizer
        )
        return query_engine.query(query_str)
