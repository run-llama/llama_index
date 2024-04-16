import asyncio
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.data_structs import IndexList
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.embeddings.utils import EmbedType, resolve_embed_model
from llama_index.core.callbacks import CallbackManager
from llama_index.core.indices.base import BaseIndex
from llama_index.core.indices.labelled_property_graph.transformations import (
    ExtractTripletsFromText,
    ExtractTripletsFromNodeRelations,
)
from llama_index.core.indices.utils import embed_nodes, async_embed_nodes
from llama_index.core.ingestion.pipeline import (
    run_transformations,
    arun_transformations,
)
from llama_index.core.graph_stores.types import (
    Entity,
    Relation,
    LabelledPropertyGraphStore,
    TRIPLET_SOURCE_KEY,
)
from llama_index.core.storage.docstore.types import RefDocInfo
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.schema import BaseNode, TransformComponent
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
        self._llm = llm or Settings.llm
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
            ExtractTripletsFromNodeRelations(),
            ExtractTripletsFromText(llm=self._llm),
        ]
        self._use_async = use_async

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
        if self._use_async:
            nodes = asyncio.run(
                arun_transformations(
                    nodes, self._kg_transformations, show_progress=self._show_progress
                )
            )

        nodes = run_transformations(
            nodes, self._kg_transformations, show_progress=self._show_progress
        )

        triplets = []
        for node in nodes:
            # remove triplets from metadata and store them separately
            node_triplets = node.metadata.pop("triplets", [])

            # add node to graph store, with id_ in metadata
            metadata = node.metadata.copy()
            metadata[TRIPLET_SOURCE_KEY] = node.id_

            for triplet in node_triplets:
                subj = Entity(text=triplet[0], properties=metadata)
                rel = Relation(text=triplet[1], properties=metadata)
                obj = Entity(text=triplet[2], properties=metadata)
                triplets.append((subj, rel, obj))

        self.lpg_graph_store.upsert_triplets(triplets)

        # add nodes to graph store -- with embeddings if needed
        if self._embed_model and self.lpg_graph_store.supports_vector_queries:
            nodes_by_id = {node.id_: node for node in nodes}
            if self._use_async:
                embed_map = asyncio.run(
                    async_embed_nodes(
                        nodes, self._embed_model, show_progress=self._show_progress
                    )
                )
            else:
                embed_map = embed_nodes(
                    nodes, self._embed_model, show_progress=self._show_progress
                )

            for node_id, embedding in embed_map.items():
                nodes_by_id[node_id].embedding = embedding

        self.lpg_graph_store.upsert_nodes(nodes)

        return nodes

    def _build_index_from_nodes(self, nodes: Optional[Sequence[BaseNode]]) -> IndexList:
        """Build index from nodes."""
        nodes = self._insert_nodes(nodes or [])
        return IndexList(nodes=[node.id_ for node in nodes])

    def as_retriever(self, include_text: bool = True, **kwargs: Any) -> BaseRetriever:
        from llama_index.core.indices.labelled_property_graph.retriever import (
            LPGRetriever,
        )
        from llama_index.core.indices.labelled_property_graph.sub_retrievers.vector_retriever import (
            LPGVectorRetriever,
        )
        from llama_index.core.indices.labelled_property_graph.sub_retrievers.llm_synonym_retriever import (
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
