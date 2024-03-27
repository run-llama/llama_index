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


class LabelledPropertyGraphIndex(BaseIndex[IndexList]):
    """An index for a labelled property graph."""

    index_struct_cls = IndexList

    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        llm: Optional[BaseLLM] = None,
        kg_transformations: Optional[List[TransformComponent]] = None,
        # vector store index params
        use_async: bool = False,
        embed_model: Optional[EmbedType] = None,
        # parent class params
        index_struct: Optional[IndexList] = None,
        storage_context: Optional[StorageContext] = None,
        callback_manager: Optional[CallbackManager] = None,
        transformations: Optional[List[TransformComponent]] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        self._llm = llm or Settings.llm
        self._embed_model = (
            resolve_embed_model(embed_model) if embed_model else Settings.embed_model
        )
        self._kg_transformations = kg_transformations or [
            ExtractTripletsFromText(llm=self._llm)
        ]
        self._use_async = use_async

        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            storage_context=storage_context,
            callback_manager=callback_manager,
            transformations=transformations,
            show_progress=show_progress,
            **kwargs,
        )

    def _build_index_from_nodes(self, nodes: Optional[Sequence[BaseNode]]) -> IndexList:
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
        triplets = []
        for node in nodes:
            node_triplets = node.metadata.pop("triplets", [])
            for triplet in node_triplets:
                subj = Entity(name=triplet[0], properties=node.metadata)
                rel = Relation(name=triplet[1], properties=node.metadata)
                obj = Entity(name=triplet[2], properties=node.metadata)
                triplets.append((subj, rel, obj))
        self._storage_context.lpg_graph_store.upsert_triplets(triplets)
        self._storage_context.docstore.add_documents(nodes)

        index_struct = self.index_struct_cls()
        for node in nodes:
            index_struct.add_node(node)
        self._storage_context.index_store.add_index_struct(index_struct)

        # create/embed tiny nodes for vector store
        # TODO add batch async?
        tiny_nodes = [
            TextNode(id_=node.id_, text=str(triplet), metadata=node.metadata)
            for node, triplets in zip(nodes, triplets)
            for triplet in triplets
        ]
        texts = [node.get_content(metadata_mode="embed") for node in tiny_nodes]
        embeddings = self._embed_model.get_text_embedding_batch(texts)

        for node, embedding in zip(tiny_nodes, embeddings):
            node.embedding = embedding

        self._storage_context.vector_store.add(tiny_nodes)

        return index_struct

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        raise NotImplementedError(
            "Retriever not implemented for LabelledPropertyGraphIndex."
        )

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
