from typing import Any, List, Optional
import os
import shutil
import time
import logging

from click import clear
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


class ObjectBoxVectorStore(BasePydanticVectorStore):
    """
    ObjectBox VectorStore For LlamaIndex
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
            **data
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
                entity_id = self._box.put(
                    self._entity_class(
                        text=node.get_content(metadata_mode=MetadataMode.NONE),
                        metadata=node.metadata,
                        embeddings=node.embedding,
                    )
                )
                ids.append(str(entity_id))
            if self.do_log:
                end = time.perf_counter()
                _logger.info(
                    f"ObjectBox stored {len(ids)} documents in {end - start} seconds"
                )
            return ids


    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        try:
            self._box.remove(int(ref_doc_id))
        except ValueError:
            raise ValueError(f"Invalid doc id: {ref_doc_id}")


    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        # TODO: Check if filters can be used to implement
        #       conditional filters
        if node_ids is not None:
            for node_id in node_ids:
                self._box.remove(int(node_id))


    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        # TODO: Check if filters can be used to implement
        #       conditional filters
        if node_ids is not None:
            retrieved_nodes: list[BaseNode] = []
            with self._store.read_tx():
                for node_id in node_ids:
                    try:
                        entity = self._box.get(int(node_id))
                        if entity is None:
                            _logger.info(f"No entity with id = {int(node_id)} was found")
                            continue
                        retrieved_nodes.append(
                            TextNode(
                                text=entity.text,
                                id_=str(entity.id),
                                metadata=entity.metadata,
                            )
                        )
                    except ValueError:
                        raise ValueError(f"Invalid node id: {node_id}")
                return retrieved_nodes
        else:
            raise ValueError("node_ids cannot be None")


    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
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
            node = TextNode(text=entity.text, id_=str(entity.id), metadata=entity.metadata)
            ids.append(str(entity.id))
            nodes.append(node)
            similarities.append(score)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)


    def clear(self) -> None:
        self._box.remove_all()


    def close(self):
        self._store.close()


    def _create_entity_class(self) -> Entity:
        """Dynamically define an Entity class according to the parameters."""

        @Entity()
        class VectorEntity:
            id = Id()
            text = String()
            metadata = Property(dict, type=PropertyType.flex, id=3, uid=1003)
            embeddings = Float32Vector(
                index=HnswIndex(
                    dimensions=self.embedding_dimensions,
                    distance_type=self.distance_type,
                )
            )

        return VectorEntity

    def _create_box_store(self) -> Store:
        """registering the VectorEntity model and setting up objectbox database"""
        db_path = DIRECTORY if self.db_directory is None else self.db_directory
        if self.clear_db and os.path.exists(db_path):
            shutil.rmtree(db_path)
        model = Model()
        model.entity(self._entity_class)
        return Store(model=model, directory=db_path)
