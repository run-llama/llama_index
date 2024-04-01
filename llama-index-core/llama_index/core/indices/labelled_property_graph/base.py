import asyncio
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.data_structs import IndexDict
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.embeddings.utils import EmbedType, resolve_embed_model
from llama_index.core.callbacks import CallbackManager
from llama_index.core.indices.base import BaseIndex
from llama_index.core.indices.labelled_property_graph.transformations import (
    ExtractTripletsFromText,
)
from llama_index.core.ingestion.pipeline import (
    run_transformations,
    arun_transformations,
)
from llama_index.core.graph_stores.types import Entity, Relation
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.storage.docstore.types import RefDocInfo
from llama_index.core.schema import BaseNode, TextNode, TransformComponent
from llama_index.core.settings import Settings


class LabelledPropertyGraphIndex(BaseIndex[IndexDict]):
    """An index for a labelled property graph."""

    index_struct_cls = IndexDict

    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        llm: Optional[BaseLLM] = None,
        kg_transformations: Optional[List[TransformComponent]] = None,
        # vector store index params
        use_async: bool = True,
        embed_model: Optional[EmbedType] = None,
        embed_triplets: bool = True,
        # parent class params
        index_struct: Optional[IndexDict] = None,
        storage_context: Optional[StorageContext] = None,
        callback_manager: Optional[CallbackManager] = None,
        transformations: Optional[List[TransformComponent]] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        self._llm = llm or Settings.llm

        if embed_triplets:
            self._embed_model = (
                resolve_embed_model(embed_model)
                if embed_model
                else Settings.embed_model
            )
        else:
            self._embed_model = None

        self._kg_transformations = kg_transformations or [
            ExtractTripletsFromText(llm=self._llm)
        ]
        self._use_async = use_async

        self.lpg_graph_store = storage_context.lpg_graph_store
        self.vector_store = storage_context.vector_store

        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            storage_context=storage_context,
            callback_manager=callback_manager,
            transformations=transformations,
            show_progress=show_progress,
            **kwargs,
        )

    def _build_index_from_nodes(self, nodes: Optional[Sequence[BaseNode]]) -> IndexDict:
        """Build index from nodes."""
        if self._use_async:
            nodes = asyncio.run(
                arun_transformations(
                    nodes, self._kg_transformations, show_progress=self._show_progress
                )
            )

        nodes = run_transformations(
            nodes, self._kg_transformations, show_progress=self._show_progress
        )

        # TODO: this is way to complicated. Can we simplify this?
        # One idea -- only use a graph store, don't use a vector store, no in-memory simple.
        triplets = []
        for node in nodes:
            # remove triplets from metadata and store them separately
            node_triplets = node.metadata.pop("triplets", [])

            # add node to graph store, with id_ in metadata
            node.metadata["id_"] = node.id_

            for triplet in node_triplets:
                subj = Entity(name=triplet[0], properties=node.metadata)
                rel = Relation(name=triplet[1], properties=node.metadata)
                obj = Entity(name=triplet[2], properties=node.metadata)
                triplets.append((subj, rel, obj))

        self.lpg_graph_store.upsert_triplets(triplets)

        if not self.lpg_graph_store.supports_nodes:
            self.docstore.add_documents(nodes)

            index_struct = self.index_struct_cls()
            for node in nodes:
                index_struct.add_node(node)
            self._storage_context.index_store.add_index_struct(index_struct)

        if not self.lpg_graph_store.supports_vectors and self._embed_model is not None:
            # create/embed tiny nodes for vector store
            # TODO add batch async?
            tiny_nodes = []
            for node in nodes:
                node_triplets = [
                    x for x in triplets if node.id_ in x[0].properties["id_"]
                ]
                tiny_nodes.extend(
                    [
                        TextNode(
                            text=f"{triplet[0].name}, {triplet[1].name}, {triplet[2].name}",
                            metadata=node.metadata,
                        )
                        for triplet in node_triplets
                    ]
                )

            texts = [node.get_content(metadata_mode="embed") for node in tiny_nodes]
            embeddings = self._embed_model.get_text_embedding_batch(texts)

            for node, embedding in zip(tiny_nodes, embeddings):
                node.embedding = embedding

            self.vector_store.add(tiny_nodes)

            if not self.vector_store.stores_text:
                for node in tiny_nodes:
                    index_struct.add_node(node)
                self.docstore.add_documents(tiny_nodes)
                self._storage_context.index_store.add_index_struct(index_struct)

        return index_struct

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
            LPGVectorRetriever(
                index=self,
                include_text=include_text,
                embed_model=self._embed_model,
                **kwargs,
            ),
            LLMSynonymRetriever(
                index=self,
                include_text=include_text,
                llm=self._llm,
                show_progress=self._show_progress,
                **kwargs,
            ),
        ]

        return LPGRetriever(retrievers, use_async=self._use_async, **kwargs)

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        """Delete a node."""
        raise NotImplementedError(
            "Delete not implemented for LabelledPropertyGraphIndex."
        )

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Index-specific logic for inserting nodes to the index struct."""
        raise NotImplementedError(
            "Insert not implemented for LabelledPropertyGraphIndex."
        )

    def ref_doc_info(self) -> Dict[str, RefDocInfo]:
        """Retrieve a dict mapping of ingested documents and their nodes+metadata."""
        raise NotImplementedError(
            "Ref doc info not implemented for LabelledPropertyGraphIndex."
        )
