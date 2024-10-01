from llama_index.core.storage.kvstore.types import (
    BaseKVStore,
)
from typing import Dict, List, Optional, Tuple
from couchbase.options import ClusterOptions
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.exceptions import DocumentNotFoundException
from couchbase.kv_range_scan import RangeScan

from acouchbase.cluster import Cluster as AsyncCluster
from datetime import timedelta

DEFAULT_COLLECTION = "data"
DEFAULT_BATCH_SIZE = 1


class CouchbaseKVStore(BaseKVStore):
    def __init__(
        self,
        connection_string: str,
        db_username: str,
        db_password: str,
        bucket_name: str,
        scope_name: str,
    ) -> None:
        """Init a CouchbaseKVStore."""
        auth = PasswordAuthenticator(db_username, db_password)
        self._cluster = Cluster(connection_string, ClusterOptions(auth))

        # wait until the cluster is ready
        self._cluster.wait_until_ready(timedelta(seconds=5))

        # create the async cluster object
        self._acluster = AsyncCluster(
            connection_string, ClusterOptions(PasswordAuthenticator)
        )

        self._bucketname = bucket_name
        self._scopename = scope_name

        # Create references to the bucket, scope, and collection
        self._bucket = self._cluster.bucket(bucket_name)
        self._scope = self._bucket.scope(scope_name)

        self._abucket = self._acluster.bucket(bucket_name)
        self._ascope = self._abucket.scope(scope_name)

    def put(self, key: str, val: dict, collection: str = DEFAULT_COLLECTION) -> None:
        db_collection = self._scope.collection(collection)
        db_collection.upsert(key, val)

    async def aput(
        self, key: str, val: dict, collection: str = DEFAULT_COLLECTION
    ) -> None:
        db_collection = self._ascope.collection(collection)
        await db_collection.upsert(key, val)

    def put_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        db_collection = self._scope.collection(collection)
        batches = [
            kv_pairs[i : i + batch_size] for i in range(0, len(kv_pairs), batch_size)
        ]

        for batch in batches:
            docs = dict(batch)
            db_collection.upsert_multi(batch)

    async def aput_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        # by default, support a batch size of 1
        if batch_size != 1:
            raise NotImplementedError("Batching not supported by this key-value store.")
        else:
            for key, val in kv_pairs:
                await self.aput(key, val, collection=collection)

    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        try:
            db_collection = self._scope.collection(collection)
        except DocumentNotFoundException:
            return None
        return db_collection.get(key).content_as[dict]

    async def aget(
        self, key: str, collection: str = DEFAULT_COLLECTION
    ) -> Optional[dict]:
        db_collection = self._ascope.collection(collection)
        try:
            return (await db_collection.get(key)).content_as[dict]
        except DocumentNotFoundException:
            return None

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        output = {}
        db_collection = self._scope.collection(collection)
        results = db_collection.scan(RangeScan())

        for result in results:
            output[result.id] = result.content_as[dict]

        return output

    async def aget_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        output = {}
        db_collection = self._ascope.collection(collection)
        results = db_collection.scan(RangeScan())
        async for result in results:
            output[result.id] = result.content_as[dict]

        return output

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        db_collection = self._scope.collection(collection)
        try:
            db_collection.remove(key)
            return True
        except DocumentNotFoundException:
            return False

    async def adelete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        db_collection = self._ascope.collection(collection)
        try:
            await db_collection.remove(key)
            return True
        except DocumentNotFoundException:
            return False
