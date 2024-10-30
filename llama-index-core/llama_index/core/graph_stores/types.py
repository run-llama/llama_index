import fsspec
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Set,
    Sequence,
    Protocol,
    runtime_checkable,
)

from llama_index.core.bridge.pydantic import BaseModel, Field, SerializeAsAny
from llama_index.core.graph_stores.prompts import DEFAULT_CYPHER_TEMPALTE
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from llama_index.core.vector_stores.types import VectorStoreQuery

DEFAULT_PERSIST_DIR = "./storage"
DEFAULT_PERSIST_FNAME = "graph_store.json"
DEFUALT_PG_PERSIST_FNAME = "property_graph_store.json"

TRIPLET_SOURCE_KEY = "triplet_source_id"
VECTOR_SOURCE_KEY = "vector_source_id"
KG_NODES_KEY = "nodes"
KG_RELATIONS_KEY = "relations"
KG_SOURCE_REL = "SOURCE"


class LabelledNode(BaseModel):
    """An entity in a graph."""

    label: str = Field(default="node", description="The label of the node.")
    embedding: Optional[List[float]] = Field(
        default=None, description="The embeddings of the node."
    )
    properties: Dict[str, Any] = Field(default_factory=dict)

    @abstractmethod
    def __str__(self) -> str:
        """Return the string representation of the node."""
        ...

    @property
    @abstractmethod
    def id(self) -> str:
        """Get the node id."""
        ...


class EntityNode(LabelledNode):
    """An entity in a graph."""

    name: str = Field(description="The name of the entity.")
    label: str = Field(default="entity", description="The label of the node.")
    properties: Dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        """Return the string representation of the node."""
        if self.properties:
            return f"{self.name} ({self.properties})"
        return self.name

    @property
    def id(self) -> str:
        """Get the node id."""
        return self.name.replace('"', " ")


class ChunkNode(LabelledNode):
    """A text chunk in a graph."""

    text: str = Field(description="The text content of the chunk.")
    id_: Optional[str] = Field(
        default=None, description="The id of the node. Defaults to a hash of the text."
    )
    label: str = Field(default="text_chunk", description="The label of the node.")
    properties: Dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        """Return the string representation of the node."""
        return self.text

    @property
    def id(self) -> str:
        """Get the node id."""
        return str(hash(self.text)) if self.id_ is None else self.id_


class Relation(BaseModel):
    """A relation connecting two entities in a graph."""

    label: str
    source_id: str
    target_id: str
    properties: Dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        """Return the string representation of the relation."""
        if self.properties:
            return f"{self.label} ({self.properties})"
        return self.label

    @property
    def id(self) -> str:
        """Get the relation id."""
        return self.label


Triplet = Tuple[LabelledNode, Relation, LabelledNode]


class LabelledPropertyGraph(BaseModel):
    """In memory labelled property graph containing entities and relations."""

    nodes: SerializeAsAny[Dict[str, LabelledNode]] = Field(default_factory=dict)
    relations: SerializeAsAny[Dict[str, Relation]] = Field(default_factory=dict)
    triplets: Set[Tuple[str, str, str]] = Field(
        default_factory=set, description="List of triplets (subject, relation, object)."
    )

    def _get_relation_key(
        self,
        relation: Optional[Relation] = None,
        subj_id: Optional[str] = None,
        obj_id: Optional[str] = None,
        rel_id: Optional[str] = None,
    ) -> str:
        """Get relation id."""
        if relation:
            return f"{relation.source_id}_{relation.label}_{relation.target_id}"
        return f"{subj_id}_{rel_id}_{obj_id}"

    def get_all_nodes(self) -> List[LabelledNode]:
        """Get all entities."""
        return list(self.nodes.values())

    def get_all_relations(self) -> List[Relation]:
        """Get all relations."""
        return list(self.relations.values())

    def get_triplets(self) -> List[Triplet]:
        """Get all triplets."""
        return [
            (
                self.nodes[subj],
                self.relations[
                    self._get_relation_key(obj_id=obj, subj_id=subj, rel_id=rel)
                ],
                self.nodes[obj],
            )
            for subj, rel, obj in self.triplets
        ]

    def add_triplet(self, triplet: Triplet) -> None:
        """Add a triplet."""
        subj, rel, obj = triplet
        if (subj.id, rel.id, obj.id) in self.triplets:
            return

        self.triplets.add((subj.id, rel.id, obj.id))
        self.nodes[subj.id] = subj
        self.nodes[obj.id] = obj
        self.relations[self._get_relation_key(relation=rel)] = rel

    def add_node(self, node: LabelledNode) -> None:
        """Add a node."""
        self.nodes[node.id] = node

    def add_relation(self, relation: Relation) -> None:
        """Add a relation."""
        if relation.source_id not in self.nodes:
            self.nodes[relation.source_id] = EntityNode(name=relation.source_id)
        if relation.target_id not in self.nodes:
            self.nodes[relation.target_id] = EntityNode(name=relation.target_id)

        self.add_triplet(
            (self.nodes[relation.source_id], relation, self.nodes[relation.target_id])
        )

    def delete_triplet(self, triplet: Triplet) -> None:
        """Delete a triplet."""
        subj, rel, obj = triplet
        if (subj.id, rel.id, obj.id) not in self.triplets:
            return

        self.triplets.remove((subj.id, rel.id, obj.id))
        if subj.id in self.nodes:
            del self.nodes[subj.id]
        if obj.id in self.nodes:
            del self.nodes[obj.id]

        rel_key = self._get_relation_key(relation=rel)
        if rel_key in self.relations:
            del self.relations[rel_key]

    def delete_node(self, node: LabelledNode) -> None:
        """Delete a node."""
        if node.id in self.nodes:
            del self.nodes[node.id]

    def delete_relation(self, relation: Relation) -> None:
        """Delete a relation."""
        rel_key = self._get_relation_key(relation=relation)
        if rel_key in self.relations:
            del self.relations[rel_key]


@runtime_checkable
class GraphStore(Protocol):
    """Abstract graph store protocol.

    This protocol defines the interface for a graph store, which is responsible
    for storing and retrieving knowledge graph data.

    Attributes:
        client: Any: The client used to connect to the graph store.
        get: Callable[[str], List[List[str]]]: Get triplets for a given subject.
        get_rel_map: Callable[[Optional[List[str]], int], Dict[str, List[List[str]]]]:
            Get subjects' rel map in max depth.
        upsert_triplet: Callable[[str, str, str], None]: Upsert a triplet.
        delete: Callable[[str, str, str], None]: Delete a triplet.
        persist: Callable[[str, Optional[fsspec.AbstractFileSystem]], None]:
            Persist the graph store to a file.
        get_schema: Callable[[bool], str]: Get the schema of the graph store.
    """

    schema: str = ""

    @property
    def client(self) -> Any:
        """Get client."""
        ...

    def get(self, subj: str) -> List[List[str]]:
        """Get triplets."""
        ...

    def get_rel_map(
        self, subjs: Optional[List[str]] = None, depth: int = 2, limit: int = 30
    ) -> Dict[str, List[List[str]]]:
        """Get depth-aware rel map."""
        ...

    def upsert_triplet(self, subj: str, rel: str, obj: str) -> None:
        """Add triplet."""
        ...

    def delete(self, subj: str, rel: str, obj: str) -> None:
        """Delete triplet."""
        ...

    def persist(
        self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> None:
        """Persist the graph store to a file."""
        return

    def get_schema(self, refresh: bool = False) -> str:
        """Get the schema of the graph store."""
        ...

    def query(self, query: str, param_map: Optional[Dict[str, Any]] = {}) -> Any:
        """Query the graph store with statement and parameters."""
        ...


class PropertyGraphStore(ABC):
    """Abstract labelled graph store protocol.

    This protocol defines the interface for a graph store, which is responsible
    for storing and retrieving knowledge graph data.

    Attributes:
        client: Any: The client used to connect to the graph store.
        get: Callable[[str], List[List[str]]]: Get triplets for a given subject.
        get_rel_map: Callable[[Optional[List[str]], int], Dict[str, List[List[str]]]]:
            Get subjects' rel map in max depth.
        upsert_triplet: Callable[[str, str, str], None]: Upsert a triplet.
        delete: Callable[[str, str, str], None]: Delete a triplet.
        persist: Callable[[str, Optional[fsspec.AbstractFileSystem]], None]:
            Persist the graph store to a file.
    """

    supports_structured_queries: bool = False
    supports_vector_queries: bool = False
    text_to_cypher_template: PromptTemplate = DEFAULT_CYPHER_TEMPALTE

    @property
    def client(self) -> Any:
        """Get client."""
        ...

    @abstractmethod
    def get(
        self,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[LabelledNode]:
        """Get nodes with matching values."""
        ...

    @abstractmethod
    def get_triplets(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Get triplets with matching values."""
        ...

    @abstractmethod
    def get_rel_map(
        self,
        graph_nodes: List[LabelledNode],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Get depth-aware rel map."""
        ...

    def get_llama_nodes(self, node_ids: List[str]) -> List[BaseNode]:
        """Get llama-index nodes."""
        nodes = self.get(ids=node_ids)
        converted_nodes = []
        for node in nodes:
            try:
                converted_nodes.append(metadata_dict_to_node(node.properties))
                converted_nodes[-1].set_content(node.text)  # type: ignore
            except Exception:
                continue

        return converted_nodes

    @abstractmethod
    def upsert_nodes(self, nodes: Sequence[LabelledNode]) -> None:
        """Upsert nodes."""
        ...

    @abstractmethod
    def upsert_relations(self, relations: List[Relation]) -> None:
        """Upsert relations."""
        ...

    def upsert_llama_nodes(self, llama_nodes: List[BaseNode]) -> None:
        """Add llama-index nodes."""
        converted_nodes = []
        for llama_node in llama_nodes:
            metadata_dict = node_to_metadata_dict(llama_node, remove_text=True)
            converted_nodes.append(
                ChunkNode(
                    text=llama_node.get_content(metadata_mode=MetadataMode.NONE),
                    id_=llama_node.id_,
                    properties=metadata_dict,
                    embedding=llama_node.embedding,
                )
            )
        self.upsert_nodes(converted_nodes)

    @abstractmethod
    def delete(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Delete matching data."""
        ...

    def delete_llama_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        ref_doc_ids: Optional[List[str]] = None,
    ) -> None:
        """Delete llama-index nodes.

        Intended to delete any nodes in the graph store associated
        with the given llama-index node_ids or ref_doc_ids.
        """
        nodes = []

        node_ids = node_ids or []
        for id_ in node_ids:
            nodes.extend(self.get(properties={TRIPLET_SOURCE_KEY: id_}))

        if len(node_ids) > 0:
            nodes.extend(self.get(ids=node_ids))

        ref_doc_ids = ref_doc_ids or []
        for id_ in ref_doc_ids:
            nodes.extend(self.get(properties={"ref_doc_id": id_}))

        if len(ref_doc_ids) > 0:
            nodes.extend(self.get(ids=ref_doc_ids))

        self.delete(ids=[node.id for node in nodes])

    @abstractmethod
    def structured_query(
        self, query: str, param_map: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Query the graph store with statement and parameters."""
        ...

    @abstractmethod
    def vector_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> Tuple[List[LabelledNode], List[float]]:
        """Query the graph store with a vector store query."""
        ...

    def persist(
        self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> None:
        """Persist the graph store to a file."""
        return

    def get_schema(self, refresh: bool = False) -> Any:
        """Get the schema of the graph store."""
        return None

    def get_schema_str(self, refresh: bool = False) -> str:
        """Get the schema of the graph store as a string."""
        return str(self.get_schema(refresh=refresh))

    ### ----- Async Methods ----- ###

    async def aget(
        self,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[LabelledNode]:
        """Asynchronously get nodes with matching values."""
        return self.get(properties, ids)

    async def aget_triplets(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Asynchronously get triplets with matching values."""
        return self.get_triplets(entity_names, relation_names, properties, ids)

    async def aget_rel_map(
        self,
        graph_nodes: List[LabelledNode],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Asynchronously get depth-aware rel map."""
        return self.get_rel_map(graph_nodes, depth, limit, ignore_rels)

    async def aget_llama_nodes(self, node_ids: List[str]) -> List[BaseNode]:
        """Asynchronously get nodes."""
        nodes = await self.aget(ids=node_ids)
        converted_nodes = []
        for node in nodes:
            try:
                converted_nodes.append(metadata_dict_to_node(node.properties))
                converted_nodes[-1].set_content(node.text)  # type: ignore
            except Exception:
                continue

        return converted_nodes

    async def aupsert_nodes(self, nodes: List[LabelledNode]) -> None:
        """Asynchronously add nodes."""
        return self.upsert_nodes(nodes)

    async def aupsert_relations(self, relations: List[Relation]) -> None:
        """Asynchronously add relations."""
        return self.upsert_relations(relations)

    async def adelete(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Asynchronously delete matching data."""
        return self.delete(entity_names, relation_names, properties, ids)

    async def adelete_llama_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        ref_doc_ids: Optional[List[str]] = None,
    ) -> None:
        """Asynchronously delete llama-index nodes."""
        return self.delete_llama_nodes(node_ids, ref_doc_ids)

    async def astructured_query(
        self, query: str, param_map: Optional[Dict[str, Any]] = {}
    ) -> Any:
        """Asynchronously query the graph store with statement and parameters."""
        return self.structured_query(query, param_map)

    async def avector_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> Tuple[List[LabelledNode], List[float]]:
        """Asynchronously query the graph store with a vector store query."""
        return self.vector_query(query, **kwargs)

    async def aget_schema(self, refresh: bool = False) -> str:
        """Asynchronously get the schema of the graph store."""
        return self.get_schema(refresh=refresh)

    async def aget_schema_str(self, refresh: bool = False) -> str:
        """Asynchronously get the schema of the graph store as a string."""
        return str(await self.aget_schema(refresh=refresh))
