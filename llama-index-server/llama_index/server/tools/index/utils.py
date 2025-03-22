from datetime import timedelta

from cachetools import TTLCache, cached  # type: ignore

from llama_index.core.storage import StorageContext


@cached(
    TTLCache(maxsize=10, ttl=timedelta(minutes=5).total_seconds()),
    key=lambda *args, **kwargs: "global_storage_context",
)
def get_storage_context(persist_dir: str) -> StorageContext:
    return StorageContext.from_defaults(persist_dir=persist_dir)
