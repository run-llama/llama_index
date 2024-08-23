"""Knowledge Graph Index.

Build a KG by extracting triplets, and leveraging the KG during query-time.

"""

import logging
import deprecated
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.constants import GRAPH_STORE_KEY
from llama_index.core.data_structs.data_structs import KG
from llama_index.core.graph_stores.simple import SimpleGraphStore
from llama_index.core.graph_stores.types import GraphStore
from llama_index.core.indices.base import BaseIndex
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.default_prompts import (
    DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
)
from llama_index.core.schema import BaseNode, IndexNode, MetadataMode
from llama_index.core.settings import Settings
from llama_index.core.storage.docstore.types import RefDocInfo
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.utils import get_tqdm_iterable

logger = logging.getLogger(__name__)


@deprecated.deprecated(
    version="0.10.53",
    reason=(
        "The KnowledgeGraphIndex class has been deprecated. "
        "Please use the new PropertyGraphIndex class instead. "
        "If a certain graph store integration is missing in the new class, "
        "please open an issue on the GitHub repository or contribute it!"
    ),
)
class KnowledgeGraphIndex(BaseIndex[KG]):
    """Knowledge Graph Index.

    Build a KG by extracting triplets, and leveraging the KG during query-time.

    Args:
        kg_triplet_extract_template (BasePromptTemplate): The prompt to use for
            extracting triplets.
        max_triplets_per_chunk (int): The maximum number of triplets to extract.
        storage_context (Optional[StorageContext]): The storage context to use.
        graph_store (Optional[GraphStore]): The graph store to use.
        show_progress (bool): Whether to show tqdm progress bars. Defaults to False.
        include_embeddings (bool): Whether to include embeddings in the index.
            Defaults to False.
        max_object_length (int): The maximum length of the object in a triplet.
            Defaults to 128.
        kg_triplet_extract_fn (Optional[Callable]): The function to use for
            extracting triplets. Defaults to None.

    """

    index_struct_cls = KG

    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        objects: Optional[Sequence[IndexNode]] = None,
        index_struct: Optional[KG] = None,
        llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,
        storage_context: Optional[StorageContext] = None,
        kg_triplet_extract_template: Optional[BasePromptTemplate] = None,
        max_triplets_per_chunk: int = 10,
        include_embeddings: bool = False,
        show_progress: bool = False,
        max_object_length: int = 128,
        kg_triplet_extract_fn: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        # need to set parameters before building index in base class.
        self.include_embeddings = include_embeddings
        self.max_triplets_per_chunk = max_triplets_per_chunk
        self.kg_triplet_extract_template = (
            kg_triplet_extract_template or DEFAULT_KG_TRIPLET_EXTRACT_PROMPT
        )
        # NOTE: Partially format keyword extract template here.
        self.kg_triplet_extract_template = (
            self.kg_triplet_extract_template.partial_format(
                max_knowledge_triplets=self.max_triplets_per_chunk
            )
        )
        self._max_object_length = max_object_length
        self._kg_triplet_extract_fn = kg_triplet_extract_fn

        self._llm = llm or Settings.llm
        self._embed_model = embed_model or Settings.embed_model

        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            storage_context=storage_context,
            show_progress=show_progress,
            objects=objects,
            **kwargs,
        )

        # TODO: legacy conversion - remove in next release
        if (
            len(self.index_struct.table) > 0
            and isinstance(self.graph_store, SimpleGraphStore)
            and len(self.graph_store._data.graph_dict) == 0
        ):
            logger.warning("Upgrading previously saved KG index to new storage format.")
            self.graph_store._data.graph_dict = self.index_struct.rel_map

    @property
    def graph_store(self) -> GraphStore:
        return self._graph_store

    def as_retriever(
        self,
        retriever_mode: Optional[str] = None,
        embed_model: Optional[BaseEmbedding] = None,
        **kwargs: Any,
    ) -> BaseRetriever:
        from llama_index.core.indices.knowledge_graph.retrievers import (
            KGRetrieverMode,
            KGTableRetriever,
        )

        if len(self.index_struct.embedding_dict) > 0 and retriever_mode is None:
            retriever_mode = KGRetrieverMode.HYBRID

        return KGTableRetriever(
            self,
            object_map=self._object_map,
            llm=self._llm,
            embed_model=embed_model or self._embed_model,
            retriever_mode=retriever_mode,
            **kwargs,
        )

    def _extract_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        if self._kg_triplet_extract_fn is not None:
            return self._kg_triplet_extract_fn(text)
        else:
            return self._llm_extract_triplets(text)

    def _llm_extract_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract keywords from text."""
        response = self._llm.predict(
            self.kg_triplet_extract_template,
            text=text,
        )
        return self._parse_triplet_response(
            response, max_length=self._max_object_length
        )

    @staticmethod
    def _parse_triplet_response(
        response: str, max_length: int = 128
    ) -> List[Tuple[str, str, str]]:
        knowledge_strs = response.strip().split("\n")
        results = []
        for text in knowledge_strs:
            if "(" not in text or ")" not in text or text.index(")") < text.index("("):
                # skip empty lines and non-triplets
                continue
            triplet_part = text[text.index("(") + 1 : text.index(")")]
            tokens = triplet_part.split(",")
            if len(tokens) != 3:
                continue

            if any(len(s.encode("utf-8")) > max_length for s in tokens):
                # We count byte-length instead of len() for UTF-8 chars,
                # will skip if any of the tokens are too long.
                # This is normally due to a poorly formatted triplet
                # extraction, in more serious KG building cases
                # we'll need NLP models to better extract triplets.
                continue

            subj, pred, obj = map(str.strip, tokens)
            if not subj or not pred or not obj:
                # skip partial triplets
                continue

            # Strip double quotes and Capitalize triplets for disambiguation
            subj, pred, obj = (
                entity.strip('"').capitalize() for entity in [subj, pred, obj]
            )

            results.append((subj, pred, obj))
        return results

    def _build_index_from_nodes(self, nodes: Sequence[BaseNode]) -> KG:
        """Build the index from nodes."""
        # do simple concatenation
        index_struct = self.index_struct_cls()
        nodes_with_progress = get_tqdm_iterable(
            nodes, self._show_progress, "Processing nodes"
        )
        for n in nodes_with_progress:
            triplets = self._extract_triplets(
                n.get_content(metadata_mode=MetadataMode.LLM)
            )
            logger.debug(f"> Extracted triplets: {triplets}")
            for triplet in triplets:
                subj, _, obj = triplet
                self.upsert_triplet(triplet)
                index_struct.add_node([subj, obj], n)

            if self.include_embeddings:
                triplet_texts = [str(t) for t in triplets]

                embed_outputs = self._embed_model.get_text_embedding_batch(
                    triplet_texts, show_progress=self._show_progress
                )
                for rel_text, rel_embed in zip(triplet_texts, embed_outputs):
                    index_struct.add_to_embedding_dict(rel_text, rel_embed)

        return index_struct

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Insert a document."""
        for n in nodes:
            triplets = self._extract_triplets(
                n.get_content(metadata_mode=MetadataMode.LLM)
            )
            logger.debug(f"Extracted triplets: {triplets}")
            for triplet in triplets:
                subj, _, obj = triplet
                triplet_str = str(triplet)
                self.upsert_triplet(triplet)
                self._index_struct.add_node([subj, obj], n)
                if (
                    self.include_embeddings
                    and triplet_str not in self._index_struct.embedding_dict
                ):
                    rel_embedding = self._embed_model.get_text_embedding(triplet_str)
                    self._index_struct.add_to_embedding_dict(triplet_str, rel_embedding)

        # Update the storage context's index_store
        self._storage_context.index_store.add_index_struct(self._index_struct)

    def upsert_triplet(
        self, triplet: Tuple[str, str, str], include_embeddings: bool = False
    ) -> None:
        """Insert triplets and optionally embeddings.

        Used for manual insertion of KG triplets (in the form
        of (subject, relationship, object)).

        Args:
            triplet (tuple): Knowledge triplet
            embedding (Any, optional): Embedding option for the triplet. Defaults to None.
        """
        self._graph_store.upsert_triplet(*triplet)
        triplet_str = str(triplet)
        if include_embeddings:
            set_embedding = self._embed_model.get_text_embedding(triplet_str)
            self._index_struct.add_to_embedding_dict(str(triplet), set_embedding)
            self._storage_context.index_store.add_index_struct(self._index_struct)

    def add_node(self, keywords: List[str], node: BaseNode) -> None:
        """Add node.

        Used for manual insertion of nodes (keyed by keywords).

        Args:
            keywords (List[str]): Keywords to index the node.
            node (Node): Node to be indexed.

        """
        self._index_struct.add_node(keywords, node)
        self._docstore.add_documents([node], allow_update=True)

    def upsert_triplet_and_node(
        self,
        triplet: Tuple[str, str, str],
        node: BaseNode,
        include_embeddings: bool = False,
    ) -> None:
        """Upsert KG triplet and node.

        Calls both upsert_triplet and add_node.
        Behavior is idempotent; if Node already exists,
        only triplet will be added.

        Args:
            keywords (List[str]): Keywords to index the node.
            node (Node): Node to be indexed.
            include_embeddings (bool): Option to add embeddings for triplets. Defaults to False

        """
        subj, _, obj = triplet
        self.upsert_triplet(triplet)
        self.add_node([subj, obj], node)
        triplet_str = str(triplet)
        if include_embeddings:
            set_embedding = self._embed_model.get_text_embedding(triplet_str)
            self._index_struct.add_to_embedding_dict(str(triplet), set_embedding)
            self._storage_context.index_store.add_index_struct(self._index_struct)

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        """Delete a node."""
        raise NotImplementedError("Delete is not supported for KG index yet.")

    @property
    def ref_doc_info(self) -> Dict[str, RefDocInfo]:
        """Retrieve a dict mapping of ingested documents and their nodes+metadata."""
        node_doc_ids_sets = list(self._index_struct.table.values())
        node_doc_ids = list(set().union(*node_doc_ids_sets))
        nodes = self.docstore.get_nodes(node_doc_ids)

        all_ref_doc_info = {}
        for node in nodes:
            ref_node = node.source_node
            if not ref_node:
                continue

            ref_doc_info = self.docstore.get_ref_doc_info(ref_node.node_id)
            if not ref_doc_info:
                continue

            all_ref_doc_info[ref_node.node_id] = ref_doc_info
        return all_ref_doc_info

    def get_networkx_graph(self, limit: int = 100) -> Any:
        """Get networkx representation of the graph structure.

        Args:
            limit (int): Number of starting nodes to be included in the graph.

        NOTE: This function requires networkx to be installed.
        NOTE: This is a beta feature.

        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "Please install networkx to visualize the graph: `pip install networkx`"
            )

        g = nx.Graph()
        subjs = list(self.index_struct.table.keys())

        # add edges
        rel_map = self._graph_store.get_rel_map(subjs=subjs, depth=1, limit=limit)

        added_nodes = set()
        for keyword in rel_map:
            for path in rel_map[keyword]:
                subj = keyword
                for i in range(0, len(path), 2):
                    if i + 2 >= len(path):
                        break

                    if subj not in added_nodes:
                        g.add_node(subj)
                        added_nodes.add(subj)

                    rel = path[i + 1]
                    obj = path[i + 2]

                    g.add_edge(subj, obj, label=rel, title=rel)
                    subj = obj
        return g

    @property
    def query_context(self) -> Dict[str, Any]:
        return {GRAPH_STORE_KEY: self._graph_store}
