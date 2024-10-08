import logging
import os
import shutil
import sys
import time
from typing import Any, List, Optional

from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores import MetadataFilters
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from objectbox import Store, Model, Box
from objectbox.model.entity import Entity
from objectbox.model.properties import (
    VectorDistanceType,
    HnswIndex,
    Id,
    Property,
    PropertyType,
    String,
    Float32Vector,
)
from objectbox.query import Query
from pydantic import PrivateAttr

# from docs.prepare_for_build import file_name

DIRECTORY = "objectbox"
_logger = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
_logger.addHandler(handler)
_logger.setLevel(logging.INFO)


class ObjectBoxVectorStore(BasePydanticVectorStore):
    """ObjectBox vector store.

    In this vector store, embeddings are stored within a ObjectBox `Box` (collection).

    During query time, the index uses ObjectBox to query for the top-K most similar nodes.

    Args:
        embedding_dimensions (int): Number of elements in the embedding to be stored
        distance_type (objectbox.model.properties.VectorDistanceType):
            Distance metric to be used for vector search
        db_directory (str): File path where ObjectBox database files will be stored
        clear_db (bool): Whether to delete any existing database on `db_directory`
        do_log (bool): enable/disable logging

    Examples:
        `pip install llama-index-vector-stores-objectbox`

        ```python
        from llama_index.vector_stores.objectbox import ObjectBoxVectorStore
        from objectbox import VectorDistanceType

        vector_store = ObjectBoxVectorStore(
            embedding_dim,
            distance_type=VectorDistanceType.COSINE,
            db_directory="obx_data",
            clear_db=False,
            do_log=True
        )
        ```
    """

    stores_text: bool = True
    embedding_dimensions: int
    distance_type: VectorDistanceType = VectorDistanceType.EUCLIDEAN
    db_directory: Optional[str] = None
    clear_db: Optional[bool] = False
    do_log: Optional[bool] = False

    _store: Store = PrivateAttr()
    _entity_class: Entity = PrivateAttr()
    _box: Box = PrivateAttr()

    def __init__(
        self,
        embedding_dimensions: int,
        distance_type: VectorDistanceType = VectorDistanceType.EUCLIDEAN,
        db_directory: Optional[str] = None,
        clear_db: Optional[bool] = False,
        do_log: Optional[bool] = False,
        **data: Any,
    ):
        super().__init__(
            embedding_dimensions=embedding_dimensions,
            distance_type=distance_type,
            db_directory=db_directory,
            clear_db=clear_db,
            do_log=do_log,
            **data,
        )
        self._entity_class = self._create_entity_class()
        self._store = self._create_box_store()

        self._box = self._store.box(self._entity_class)

    @property
    def client(self) -> Any:
        return self._box

    def add(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        ids: list[str] = []
        start = time.perf_counter()
        with self._store.write_tx():
            for node in nodes:
                if node.embedding is None:
                    _logger.info("A node with no embedding was found ")
                    continue
                self._box.put(
                    self._entity_class(
                        node_id=node.node_id,
                        doc_id=node.ref_doc_id if node.ref_doc_id is not None else "",
                        text=node.get_content(metadata_mode=MetadataMode.NONE),
                        metadata=node.metadata,
                        embeddings=node.embedding,
                    )
                )
                ids.append(node.node_id)
            if self.do_log:
                end = time.perf_counter()
                _logger.info(
                    f"ObjectBox stored {len(ids)} nodes in {end - start} seconds"
                )
            return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        self._box.query(self._entity_class.doc_id.equals(ref_doc_id)).build().remove()

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        if filters is not None:
            raise NotImplementedError(
                "ObjectBox does not yet support delete_nodes() with metadata filters - contact us if you need this feature"
            )
        if node_ids is not None:
            query_obj = self._box.query(
                self._entity_class.node_id.equals("node_id").alias("node_id")
            ).build()
            for node_id in node_ids:
                query_obj.set_parameter_alias_string("node_id", node_id)
                query_obj.remove()

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        if filters is not None:
            raise NotImplementedError(
                "ObjectBox does not yet support get_nodes() with metadata filters - contact us if you need this feature"
            )
        if node_ids is not None:
            retrieved_nodes: list[BaseNode] = []
            with self._store.read_tx():
                query_obj = self._box.query(
                    self._entity_class.node_id.equals("node_id").alias("node_id")
                ).build()
                for node_id in node_ids:
                    try:
                        query_obj.set_parameter_alias_string("node_id", node_id)
                        entities = query_obj.find()
                        if len(entities) == 0:
                            _logger.info(f"No entity with id = {node_id} was found")
                            continue
                        retrieved_nodes.append(
                            TextNode(
                                text=entities[0].text,
                                id_=entities[0].node_id,
                                metadata=entities[0].metadata,
                            )
                        )
                    except ValueError:
                        raise ValueError(f"Invalid node id: {node_id}")
                return retrieved_nodes
        else:
            raise ValueError("node_ids cannot be None")

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        if query.filters is not None:
            raise NotImplementedError(
                "ObjectBox does not yet support query() with metadata filters - contact us if you need this feature"
            )

        query_embedding = query.query_embedding
        n_results = query.similarity_top_k

        nodes: list[TextNode] = []
        similarities: list[float] = []
        ids: list[str] = []

        start = time.perf_counter()
        query: Query = self._box.query(
            self._entity_class.embeddings.nearest_neighbor(query_embedding, n_results)
        ).build()
        results: list[tuple[Entity, float]] = query.find_with_scores()
        end = time.perf_counter()

        if self.do_log:
            _logger.info(
                f"ObjectBox retrieved {len(results)} vectors in {end - start} seconds"
            )

        for entity, score in results:
            node = TextNode(
                text=entity.text, id_=entity.node_id, metadata=entity.metadata
            )
            ids.append(entity.node_id)
            nodes.append(node)
            similarities.append(score)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def count(self) -> int:
        return self._box.count()

    def clear(self) -> None:
        self._box.remove_all()

    def close(self):
        self._store.close()

    def _create_entity_class(self) -> Entity:
        """Dynamically define an Entity class according to the parameters."""

        @Entity()
        class VectorEntity:
            id = Id()
            node_id = String()
            doc_id = String()
            text = String()
            metadata = Property(dict, type=PropertyType.flex)
            embeddings = Float32Vector(
                index=HnswIndex(
                    dimensions=self.embedding_dimensions,
                    distance_type=self.distance_type,
                )
            )

        return VectorEntity

    def _create_box_store(self) -> Store:
        """Registering the VectorEntity model and setting up objectbox database."""
        db_path = DIRECTORY if self.db_directory is None else self.db_directory
        if self.clear_db and os.path.exists(db_path):
            shutil.rmtree(db_path)
        model = Model()
        model.entity(self._entity_class)
        return Store(model=model, directory=db_path)
