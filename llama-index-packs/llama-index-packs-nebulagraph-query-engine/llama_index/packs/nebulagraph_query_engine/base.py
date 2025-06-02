"""NebulaGraph Query Engine Pack."""

import os
from enum import Enum
from typing import Any, Dict, List, Optional

from llama_index.core import (
    KnowledgeGraphIndex,
    QueryBundle,
    Settings,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.retrievers import (
    BaseRetriever,
    KGTableRetriever,
    VectorIndexRetriever,
)
from llama_index.core.schema import Document, NodeWithScore
from llama_index.graph_stores.nebula import NebulaGraphStore
from llama_index.llms.openai import OpenAI


class NebulaGraphQueryEngineType(str, Enum):
    """NebulaGraph query engine type."""

    KG_KEYWORD = "keyword"
    KG_HYBRID = "hybrid"
    RAW_VECTOR = "vector"
    RAW_VECTOR_KG_COMBO = "vector_kg"
    KG_QE = "KnowledgeGraphQueryEngine"
    KG_RAG_RETRIEVER = "KnowledgeGraphRAGRetriever"


class NebulaGraphQueryEnginePack(BaseLlamaPack):
    """NebulaGraph Query Engine pack."""

    def __init__(
        self,
        username: str,
        password: str,
        ip_and_port: str,
        space_name: str,
        edge_types: str,
        rel_prop_names: str,
        tags: str,
        max_triplets_per_chunk: int,
        docs: List[Document],
        query_engine_type: Optional[NebulaGraphQueryEngineType] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        os.environ["GRAPHD_HOST"] = "127.0.0.1"
        os.environ["NEBULA_USER"] = username
        os.environ["NEBULA_PASSWORD"] = password
        os.environ["NEBULA_ADDRESS"] = (
            ip_and_port  # such as "127.0.0.1:9669" for local instance
        )

        nebulagraph_graph_store = NebulaGraphStore(
            space_name=space_name,
            edge_types=edge_types,
            rel_prop_names=rel_prop_names,
            tags=tags,
        )

        nebulagraph_storage_context = StorageContext.from_defaults(
            graph_store=nebulagraph_graph_store
        )

        # define LLM
        self.llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")
        Settings.llm = self.llm

        nebulagraph_index = KnowledgeGraphIndex.from_documents(
            documents=docs,
            storage_context=nebulagraph_storage_context,
            max_triplets_per_chunk=max_triplets_per_chunk,
            space_name=space_name,
            edge_types=edge_types,
            rel_prop_names=rel_prop_names,
            tags=tags,
            include_embeddings=True,
        )

        # create index
        vector_index = VectorStoreIndex.from_documents(docs)

        if query_engine_type == NebulaGraphQueryEngineType.KG_KEYWORD:
            # KG keyword-based entity retrieval
            self.query_engine = nebulagraph_index.as_query_engine(
                # setting to false uses the raw triplets instead of adding the text from the corresponding nodes
                include_text=False,
                retriever_mode="keyword",
                response_mode="tree_summarize",
            )

        elif query_engine_type == NebulaGraphQueryEngineType.KG_HYBRID:
            # KG hybrid entity retrieval
            self.query_engine = nebulagraph_index.as_query_engine(
                include_text=True,
                response_mode="tree_summarize",
                embedding_mode="hybrid",
                similarity_top_k=3,
                explore_global_knowledge=True,
            )

        elif query_engine_type == NebulaGraphQueryEngineType.RAW_VECTOR:
            # Raw vector index retrieval
            self.query_engine = vector_index.as_query_engine()

        elif query_engine_type == NebulaGraphQueryEngineType.RAW_VECTOR_KG_COMBO:
            from llama_index.core.query_engine import RetrieverQueryEngine

            # create custom retriever
            nebulagraph_vector_retriever = VectorIndexRetriever(index=vector_index)
            nebulagraph_kg_retriever = KGTableRetriever(
                index=nebulagraph_index, retriever_mode="keyword", include_text=False
            )
            nebulagraph_custom_retriever = CustomRetriever(
                nebulagraph_vector_retriever, nebulagraph_kg_retriever
            )

            # create response synthesizer
            nebulagraph_response_synthesizer = get_response_synthesizer(
                response_mode="tree_summarize"
            )

            # Custom combo query engine
            self.query_engine = RetrieverQueryEngine(
                retriever=nebulagraph_custom_retriever,
                response_synthesizer=nebulagraph_response_synthesizer,
            )

        elif query_engine_type == NebulaGraphQueryEngineType.KG_QE:
            # using KnowledgeGraphQueryEngine
            from llama_index.core.query_engine import KnowledgeGraphQueryEngine

            self.query_engine = KnowledgeGraphQueryEngine(
                storage_context=nebulagraph_storage_context,
                llm=self.llm,
                verbose=True,
            )

        elif query_engine_type == NebulaGraphQueryEngineType.KG_RAG_RETRIEVER:
            # using KnowledgeGraphRAGRetriever
            from llama_index.core.query_engine import RetrieverQueryEngine
            from llama_index.core.retrievers import KnowledgeGraphRAGRetriever

            nebulagraph_graph_rag_retriever = KnowledgeGraphRAGRetriever(
                storage_context=nebulagraph_storage_context,
                llm=self.llm,
                verbose=True,
            )

            self.query_engine = RetrieverQueryEngine.from_args(
                nebulagraph_graph_rag_retriever
            )

        else:
            # KG vector-based entity retrieval
            self.query_engine = nebulagraph_index.as_query_engine()

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "llm": self.llm,
            "query_engine": self.query_engine,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return self.query_engine.query(*args, **kwargs)


class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both Vector search and Knowledge Graph search."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        kg_retriever: KGTableRetriever,
        mode: str = "OR",
    ) -> None:
        """Init params."""
        self._vector_retriever = vector_retriever
        self._kg_retriever = kg_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        kg_nodes = self._kg_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        kg_ids = {n.node.node_id for n in kg_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in kg_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(kg_ids)
        else:
            retrieve_ids = vector_ids.union(kg_ids)

        return [combined_dict[rid] for rid in retrieve_ids]
