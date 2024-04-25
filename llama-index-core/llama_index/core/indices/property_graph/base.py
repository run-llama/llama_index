import asyncio
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.data_structs import IndexList
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.embeddings.utils import EmbedType, resolve_embed_model
from llama_index.core.callbacks import CallbackManager
from llama_index.core.indices.base import BaseIndex
from llama_index.core.indices.property_graph.transformations import (
    SimpleLLMTripletExtractor,
    ImplicitEdgeExtractor,
)
from llama_index.core.ingestion.pipeline import (
    run_transformations,
    arun_transformations,
)
from llama_index.core.graph_stores.types import (
    LabelledNode,
    Relation,
    LabelledPropertyGraphStore,
    TRIPLET_SOURCE_KEY,
)
from llama_index.core.storage.docstore.types import RefDocInfo
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.schema import BaseNode, MetadataMode, TransformComponent
from llama_index.core.settings import Settings


class LabelledPropertyGraphIndex(BaseIndex[IndexList]):
    """An index for a labelled property graph."""

    index_struct_cls = IndexList

    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        llm: Optional[BaseLLM] = None,
        kg_transformations: Optional[List[TransformComponent]] = None,
        lpg_graph_store: Optional[LabelledPropertyGraphStore] = None,
        # vector store index params
        use_async: bool = True,
        embed_model: Optional[EmbedType] = None,
        embed_triplets: bool = True,
        # parent class params
        callback_manager: Optional[CallbackManager] = None,
        transformations: Optional[List[TransformComponent]] = None,
        storage_context: Optional[StorageContext] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        storage_context = storage_context or StorageContext.from_defaults(
            lpg_graph_store=lpg_graph_store
        )
        if embed_triplets and storage_context.lpg_graph_store.supports_vector_queries:
            self._embed_model = (
                resolve_embed_model(embed_model)
                if embed_model
                else Settings.embed_model
            )
        else:
            self._embed_model = None

        self._kg_transformations = kg_transformations or [
            SimpleLLMTripletExtractor(llm=llm or Settings.llm),
            ImplicitEdgeExtractor(),
        ]
        self._use_async = use_async
        self._llm = llm

        super().__init__(
            nodes=nodes,
            callback_manager=callback_manager,
            storage_context=storage_context,
            transformations=transformations,
            show_progress=show_progress,
            **kwargs,
        )

    @property
    def lpg_graph_store(self) -> LabelledPropertyGraphStore:
        """Get the labelled property graph store."""
        return self.storage_context.lpg_graph_store

    def _insert_nodes(self, nodes: Sequence[BaseNode]) -> Sequence[BaseNode]:
        """Insert nodes to the index struct."""
        # run transformations on nodes to extract triplets
        if self._use_async:
            nodes = asyncio.run(
                arun_transformations(
                    nodes, self._kg_transformations, show_progress=self._show_progress
                )
            )
        else:
            nodes = run_transformations(
                nodes, self._kg_transformations, show_progress=self._show_progress
            )

        # ensure all nodes have nodes and/or relations in metadata
        assert all(
            node.metadata.get("nodes") is not None
            or node.metadata.get("relations") is not None
            for node in nodes
        )

        kg_nodes_to_insert: List[LabelledNode] = []
        kg_rels_to_insert: List[Relation] = []
        for node in nodes:
            # remove nodes and relations from metadata
            kg_nodes = node.metadata.pop("nodes", [])
            kg_rels = node.metadata.pop("relations", [])

            # add source id to properties
            for kg_node in kg_nodes:
                kg_node.properties[TRIPLET_SOURCE_KEY] = node.id_
            for kg_rel in kg_rels:
                kg_rel.properties[TRIPLET_SOURCE_KEY] = node.id_

            # add nodes and relations to insert lists
            kg_nodes_to_insert.extend(kg_nodes)
            kg_rels_to_insert.extend(kg_rels)

        # embed nodes (if needed)
        if self._embed_model and self.lpg_graph_store.supports_vector_queries:
            # embed llama-index nodes
            node_texts = [
                node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes
            ]

            if self._use_async:
                embeddings = asyncio.run(
                    self._embed_model.aget_text_embedding_batch(node_texts)
                )
            else:
                embeddings = self._embed_model.get_text_embedding_batch(node_texts)

            for node, embedding in zip(nodes, embeddings):
                node.embedding = embedding

            # embed kg nodes
            kg_node_texts = [str(kg_node) for kg_node in kg_nodes_to_insert]

            if self._use_async:
                kg_embeddings = asyncio.run(
                    self._embed_model.aget_text_embedding_batch(kg_node_texts)
                )
            else:
                kg_embeddings = self._embed_model.get_text_embedding_batch(
                    kg_node_texts
                )

            for kg_node, embedding in zip(kg_nodes_to_insert, kg_embeddings):
                kg_node.embedding = embedding

        self.lpg_graph_store.upsert_llama_nodes(nodes)
        self.lpg_graph_store.upsert_nodes(kg_nodes_to_insert)

        # important: upsert relations after nodes
        self.lpg_graph_store.upsert_relations(kg_rels_to_insert)

        return nodes

    def _build_index_from_nodes(self, nodes: Optional[Sequence[BaseNode]]) -> IndexList:
        """Build index from nodes."""
        nodes = self._insert_nodes(nodes or [])

        # this isn't really used or needed
        return IndexList(nodes=[])

    def as_retriever(self, include_text: bool = True, **kwargs: Any) -> BaseRetriever:
        from llama_index.core.indices.property_graph.retriever import (
            LPGRetriever,
        )
        from llama_index.core.indices.property_graph.sub_retrievers.vector import (
            LPGVectorRetriever,
        )
        from llama_index.core.indices.property_graph.sub_retrievers.llm_synonym import (
            LLMSynonymRetriever,
        )

        retrievers = [
            LLMSynonymRetriever(
                graph_store=self.lpg_graph_store,
                include_text=include_text,
                llm=self._llm,
                show_progress=self._show_progress,
                **kwargs,
            ),
        ]

        if self._embed_model and self.lpg_graph_store.supports_vector_queries:
            retrievers.append(
                LPGVectorRetriever(
                    graph_store=self.lpg_graph_store,
                    include_text=include_text,
                    show_progress=self._show_progress,
                    **kwargs,
                )
            )

        return LPGRetriever(retrievers, use_async=self._use_async, **kwargs)

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        """Delete a node."""
        self.lpg_graph_store.delete(node_ids=[node_id])

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Index-specific logic for inserting nodes to the index struct."""
        self._insert_nodes(nodes)

    def ref_doc_info(self) -> Dict[str, RefDocInfo]:
        """Retrieve a dict mapping of ingested documents and their nodes+metadata."""
        raise NotImplementedError(
            "Ref doc info not implemented for LabelledPropertyGraphIndex. "
            "All inserts are already upserts."
        )
