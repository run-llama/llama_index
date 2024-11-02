"""
ApertureDB vector store index.

"""

from __future__ import annotations

import logging

from pydantic import PrivateAttr
from typing_extensions import override
from llama_index.core.schema import TextNode
from typing import Any, Dict, List, Optional
from llama_index.core.vector_stores import (VectorStoreQuery, VectorStoreQueryResult,)
from llama_index.core.vector_stores.types import BasePydanticVectorStore

import json
import numpy as np
from llama_index.core.vector_stores.utils import (
    node_to_metadata_dict,
    metadata_dict_to_node,
)
from llama_index.core.schema import MetadataMode
from aperturedb.ParallelLoader import ParallelLoader

DESCRIPTOR_SET = "llamaindex"

## Defaults as defined in the Langchain-ApertureDB integration
ENGINE = "HNSW"
METRIC = "CS"

BATCHSIZE = 1000

# Prefix for properties that are in the client metadata
PROPERTY_PREFIX = "lm_"

TEXT_PROPERTY = "text"  # Property name for the text

UNIQUEID_PROPERTY = "uniqueid"  # Property name for the unique id

class ApertureDBVectorStore(BasePydanticVectorStore):
    stores_text: bool = True
    flat_metadata: bool = True
    logger: logging.Logger = logging.getLogger(__name__)

    _client = PrivateAttr()
    _execute_query = PrivateAttr()

    """ Constructor to instantiate ApertureDB vectorstore object

    Args:
            embeddings (Embeddings): Embedding function.

            descriptor_set (str, optional): Descriptor set name. Defaults to "llamaindex".

            dimensions (Optional[int], optional):   Number of dimensions of the embeddings.
                                                    Defaults to None.
            engine (str, optional): Engine to use.
                                    Defaults to "HNSW" for new descriptorsets.

            metric (str, optional): Metric to use. Defaults to "CS" for new descriptorsets.

            log_level (int, optional): Logging level. Defaults to logging.WARN.
            overwrite (bool, optional): Default set to True. Will overwrite existing descriptor set.

    """
    @override
    def __init__(self,
                #  embeddings: BaseEmbedding,
                        descriptor_set: str = DESCRIPTOR_SET,
                        dimensions: Optional[int] = None,
                        engine: Optional[str] = None,
                        metric: Optional[str] = None,
                        log_level: int = logging.WARN,
                        properties: Optional[Dict] = None,
                        overwrite:bool = True,
                        **kwargs: Any,
            ) -> None:

        # ApertureDB imports
        try:
            from aperturedb.Utils import Utils
            from aperturedb.CommonLibrary import create_connector, execute_query
            from aperturedb.Descriptors import Descriptors

        except ImportError:
            raise ImportError(
                "ApertureDB is not installed. Please install it using "
                "'pip install aperturedb'"
            )

        super().__init__(**kwargs)

        self.logger.setLevel(log_level)
        self._descriptor_set = descriptor_set
        self._dimensions = dimensions
        self._engine = engine
        self._metric = metric
        self._properties = properties
        self._overwrite = overwrite

        ## Returns a clietn for the database
        self._client = create_connector()
        self._descriptors = Descriptors(self._client)
        self._execute_query = execute_query
        self._utils = Utils(self._client)

        try:
            self._utils.status()
        except Exception:
            self.logger.exception("Failed to connect to ApertureDB")
            raise

        self._find_or_add_descriptor_set()      ## Call to find or add a descriptor set

    @classmethod
    def class_name(cls) -> str:
        return "ApertureDB"

    @property
    def client(self) -> Any:
        """Get client."""
        return self._client

    def _find_or_add_descriptor_set(self) -> None:
        descriptor_set = self._descriptor_set
        """Checks if the descriptor set exists, if not, creates it"""
        find_ds_query = [
            {
                "FindDescriptorSet": {
                    "with_name": descriptor_set,
                    "engines": True,
                    "metrics": True,
                    "dimensions": True,
                    "results": {"all_properties": True},
                }
            }
        ]
        _, response, _ = self._execute_query(client=self._client, query=find_ds_query, blobs=[])
        n_entities = (
            len(response[0]["FindDescriptorSet"]["entities"])
            if "entities" in response[0]["FindDescriptorSet"]
            else 0
        )
        assert n_entities <= 1, "Multiple descriptor sets with the same name"

        if n_entities == 1:  # Descriptor set exists already
            e = response[0]["FindDescriptorSet"]["entities"][0]
            self.logger.info(f"Descriptor set {descriptor_set} already exists")

            engines = e["_engines"]
            assert len(engines) == 1, "Only one engine is supported"

            if self._engine is None:
                self._engine = engines[0]
            elif self._engine != engines[0]:
                self.logger.error(f"Engine mismatch: {self._engine} != {engines[0]}")

            metrics = e["_metrics"]
            assert len(metrics) == 1, "Only one metric is supported"
            if self._metric is None:
                self._metric = metrics[0]
            elif self._metric != metrics[0]:
                self.logger.error(f"Metric mismatch: {self._metric} != {metrics[0]}")

            dimensions = e["_dimensions"]
            if self._dimensions is None:
                self._dimensions = dimensions
            elif self._dimensions != dimensions:
                self.logger.error(
                    f"Dimensions mismatch: {self._dimensions} != {dimensions}"
                )

            self._properties = {
                k[len(PROPERTY_PREFIX) :]: v
                for k, v in e.items()
                if k.startswith(PROPERTY_PREFIX)
            }

        else:
            self.logger.info(
                f"Descriptor set {descriptor_set} does not exist. Creating it"
            )
            if self._engine is None:
                self._engine = ENGINE
            if self._metric is None:
                self._metric = METRIC
            if self._dimensions is None:
                self._dimensions = len(self._embedding_function.get_text_embedding("test"))   ## Very well written

            properties = (
                {PROPERTY_PREFIX + k: v for k, v in self._properties.items()}
                if self._properties is not None
                else None
            )

            self._utils.add_descriptorset(
                name=descriptor_set,
                dim=self._dimensions,
                engine=self._engine,
                metric=self._metric,
                properties=properties,
            )

            # Create indexes
            self._utils.create_entity_index("_Descriptor", "_create_txn")
            self._utils.create_entity_index("_DescriptorSet", "_name")
            self._utils.create_entity_index("_Descriptor", UNIQUEID_PROPERTY)

    def add(self, nodes: List[TextNode]) -> List[str]:

        """ Adds a list of Llamaindex nodes to the vector store. Return the ids
        Args:
            texts: List of text strings
            embedding: Embeddings object as for constructing the vectorstore
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: Additional arguments to pass to the constructor
        """
        ## Overwrite the existing descriptor set
        if self._overwrite == True:
            try:
                self.delete()
            except:
                self.logger.exception("Failed to overwrite")

        ids = []
        data = []

        for node in nodes:

            metadata = node_to_metadata_dict(node,
                                            remove_text=False,
                                            flat_metadata=self.flat_metadata)

            properties = {  "uniqueid": node.node_id,
                            "text": node.get_content(metadata_mode=MetadataMode.NONE) or "",
                            "metadata": json.dumps(metadata),
                    }
            for k, v in metadata.items():
                properties[PROPERTY_PREFIX + k] = v

            command = {
                        "AddDescriptor": {
                            "set": DESCRIPTOR_SET,
                            "properties": properties,   ## I can add nodes/metadata here.
                            "if_not_found": {
                                "uniqueid": ["==", node.node_id]
                            }
                        }
                    }

            query = [command]
            blobs = [np.array(node.embedding, dtype=np.float32).tobytes()]   ## And convert the already calculated embeddings into blobs here
            data.append((query, blobs))
            ids.append(node.node_id)

        loader = ParallelLoader(self._client)
        loader.ingest(data, batchsize=BATCHSIZE)

        return ids

    def delete_vector_store(self, descriptor_set_name: str) -> None:
        self._utils.remove_descriptorset(descriptor_set_name)

    def get_descriptor_set(self) -> List[str]:
        """Return a list of existing descriptor sets in the Apeture database"""
        return self._utils.get_descriptorset_list()

    def get(self, text_id: str) -> List[float]:
        """Get embedding."""
        # return self.node_dict[text_id]
        pass

    def query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:

        """Return nodes as response."""
        ## VectorStoreQuery has query_embedding, similarity_top_k and mode
        self._descriptors.find_similar(set=self._descriptor_set,
                                 vector=query.query_embedding,
                                 k_neighbors=query.similarity_top_k,
                                 distances=True,
                                 )

        nodes = []
        ids: List[str] = []
        similarities: List[float] = []

        for d in self._descriptors:

            metadata = json.loads(d["metadata"])
            node = metadata_dict_to_node(metadata)

            text = d.get("text")

            node.set_content(text)

            nodes.append(node)
            ids.append(d.get("id"))
            similarities.append(d["_distance"])

        return VectorStoreQueryResult(nodes = nodes, ids = ids, similarities = similarities)

    def mmr_query(self, query:VectorStoreQuery) -> VectorStoreQueryResult:
        pass

    def delete(self) -> Optional[bool]:

        """Delete embeddings (if present) from the vectorstore by vector_store name.

        Returns:
            True if the deletion was successful, False otherwise
        """

        query = [
            {
                "DeleteDescriptor": {
                    "set": self._descriptor_set,
                }
            }
        ]

        result, _ = self._utils.execute(query)
        return result