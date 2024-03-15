"""Redis Ingestion Pipeline Completion pack."""


from typing import Any, Dict, List

from llama_index.core.ingestion.cache import IngestionCache
from llama_index.core.ingestion.pipeline import IngestionPipeline
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.vector_stores.redis import RedisVectorStore


class RedisIngestionPipelinePack(BaseLlamaPack):
    """Redis Ingestion Pipeline Completion pack."""

    def __init__(
        self,
        transformations: List[TransformComponent],
        hostname: str = "localhost",
        port: int = 6379,
        cache_collection_name: str = "ingest_cache",
        vector_collection_name: str = "vector_store",
        **kwargs: Any,
    ) -> None:
        """Init params."""
        self.vector_store = RedisVectorStore(
            hostname=hostname,
            port=port,
            collection_name=vector_collection_name,
        )

        self.ingest_cache = IngestionCache(
            cache=RedisCache(
                hostname=hostname,
                port=port,
            ),
            collection_name=cache_collection_name,
        )

        self.pipeline = IngestionPipeline(
            transformations=transformations,
            cache=self.ingest_cache,
            vector_store=self.vector_store,
        )

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "pipeline": self.pipeline,
            "vector_store": self.vector_store,
            "ingest_cache": self.ingest_cache,
        }

    def run(self, inputs: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        """Run the pipeline."""
        return self.pipeline.run(nodes=inputs, **kwargs)
