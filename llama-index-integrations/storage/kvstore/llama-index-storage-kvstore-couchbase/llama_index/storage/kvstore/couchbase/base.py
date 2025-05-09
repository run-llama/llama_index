import string
from llama_index.core.storage.kvstore.types import (
    BaseKVStore,
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION,
)

from typing import Any, Dict, List, Optional, Tuple
from couchbase.cluster import Cluster
from couchbase.exceptions import (
    DocumentNotFoundException,
)
from couchbase.kv_range_scan import RangeScan

from acouchbase.cluster import Cluster as AsyncCluster


class CouchbaseKVStore(BaseKVStore):
    """Couchbase Key-Value store."""

    def __init__(
        self,
        cluster: Cluster,
        bucket_name: str,
        scope_name: str,
        async_cluster: Optional[AsyncCluster] = None,
    ) -> None:
        """
        Initializes a CouchbaseKVStore.

        Args:
            cluster (Cluster): Couchbase cluster object
            bucket_name (str): Name of the bucket to use for the key-value store
            scope_name (str): Name of the scope to use for the key-value store
            async_cluster (Optional[AsyncCluster]): Async Couchbase cluster object

        """
        if not isinstance(cluster, Cluster):
            raise ValueError(
                f"cluster should be an instance of couchbase.Cluster, "
                f"got {type(cluster)}"
            )
        self._cluster = cluster
        self._acluster = None

        if async_cluster:
            if not isinstance(async_cluster, AsyncCluster):
                raise ValueError(
                    f"async_cluster should be an instance of acouchbase.Cluster, "
                    f"got {type(async_cluster)}"
                )
            self._acluster = async_cluster

        # Check if bucket exists
        if not self._check_bucket_exists(bucket_name):
            raise ValueError(
                f"Bucket {bucket_name} does not exist. "
                " Please create the bucket before using."
            )
        self._bucketname: str = bucket_name
        self._bucket = self._cluster.bucket(bucket_name)

        # Get a list of all the scopes and collections in the bucket
        self._scope_collection_map = self._list_scope_and_collections()
        if scope_name not in self._scope_collection_map:
            raise ValueError(
                f"Scope {scope_name} does not exist in bucket {self._bucketname}. "
                "Please create the scope before use."
            )
        self._scopename = scope_name
        self._scope = self._bucket.scope(scope_name)

        if self._acluster:
            self._abucket = self._acluster.bucket(bucket_name)
            self._ascope = self._abucket.scope(scope_name)

    def _check_bucket_exists(self, bucket_name) -> bool:
        """
        Check if the bucket exists in the linked Couchbase cluster.

        Returns:
            True if the bucket exists

        """
        bucket_manager = self._cluster.buckets()
        try:
            bucket_manager.get_bucket(bucket_name)
            return True
        except Exception:
            return False

    def _validate_collection_name(self, collection_name: str) -> bool:
        """
        Check if the collection name is valid for Couchbase.
        Collection names should not contain any characters other than letters, digits, underscores, percentage and hyphens.
        """
        # Only allow letters, digits, underscores, percentage and hyphens
        allowed_chars = set(string.ascii_letters + string.digits + "_-%")

        # Check if all characters in the string are in the set of allowed characters
        return all(char in allowed_chars for char in collection_name)

    def _sanitize_collection_name(self, collection_name: str) -> str:
        """
        Sanitize the collection name to remove any invalid characters.
        The unallowed characters are replaced with underscores.
        """
        # Only allow letters, digits, underscores, percentage and hyphens
        allowed_chars = set(string.ascii_letters + string.digits + "_-%")

        # Replace invalid characters with underscores
        return "".join(
            char if char in allowed_chars else "_" for char in collection_name
        )

    def _create_collection_if_not_exists(self, collection_name: str) -> None:
        """
        Create a collection in the linked Couchbase bucket if it does not exist.
        """
        if collection_name not in self._scope_collection_map[self._scopename]:
            try:
                self._bucket.collections().create_collection(
                    scope_name=self._scopename, collection_name=collection_name
                )
                self._scope_collection_map = self._list_scope_and_collections()
            except Exception as e:
                print("Error creating collection: ", e)
                raise

    def _check_async_client(self) -> None:
        """
        Check if the async client is initialized.
        """
        if not self._acluster:
            raise ValueError("CouchbaseKVStore was not initialized with async client")

    def _list_scope_and_collections(self) -> dict[str, any]:
        """
        Return the scope and collections that exist in the linked Couchbase bucket
        Returns:
           Dict[str, Any]: Dictionary of scopes and collections in the scope in the bucket.
        """
        scope_collection_map: Dict[str, Any] = {}

        # Get a list of all scopes in the bucket
        for scope in self._bucket.collections().get_all_scopes():
            scope_collection_map[scope.name] = []

            # Get a list of all the collections in the scope
            for collection in scope.collections:
                scope_collection_map[scope.name].append(collection.name)

        return scope_collection_map

    def put(self, key: str, val: dict, collection: str = DEFAULT_COLLECTION) -> None:
        """
        Insert a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name

        """
        # Create collection if it does not exist
        collection = self._sanitize_collection_name(collection)
        self._create_collection_if_not_exists(collection)

        db_collection = self._scope.collection(collection)
        db_collection.upsert(key, val)

    async def aput(
        self, key: str, val: dict, collection: str = DEFAULT_COLLECTION
    ) -> None:
        """
        Insert a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name

        """
        self._check_async_client()

        # Create collection if it does not exist
        collection = self._sanitize_collection_name(collection)
        self._create_collection_if_not_exists(collection)

        db_collection = self._ascope.collection(collection)
        await db_collection.upsert(key, val)

    def put_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """
        Insert multiple key-value pairs into the store.

        Args:
            kv_pairs (List[Tuple[str, dict]]): list of key-value pairs
            collection (str): collection name
            batch_size (int): batch size

        """
        # Create collection if it does not exist
        collection = self._sanitize_collection_name(collection)
        self._create_collection_if_not_exists(collection)

        db_collection = self._scope.collection(collection)
        # Create batches of documents to insert
        batches = [
            kv_pairs[i : i + batch_size] for i in range(0, len(kv_pairs), batch_size)
        ]

        # Insert documents in batches
        for batch in batches:
            docs = dict(batch)
            db_collection.upsert_multi(docs)

    async def aput_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """
        Insert multiple key-value pairs into the store. Note that batch_size is not supported by this key-value store for async operations.

        Args:
            kv_pairs (List[Tuple[str, dict]]): list of key-value pairs
            collection (str): collection name
            batch_size (int): batch size

        """
        # CouchbaseKVStore support only a batch size of 1 in async mode
        if batch_size != 1:
            raise NotImplementedError("Batching not supported by this key-value store.")
        else:
            for key, val in kv_pairs:
                await self.aput(key, val, collection=collection)

    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        """
        Get a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        try:
            # Create collection if it does not exist
            collection = self._sanitize_collection_name(collection)
            self._create_collection_if_not_exists(collection)

            db_collection = self._scope.collection(collection)
            document = db_collection.get(key).content_as[dict]
        except DocumentNotFoundException:
            return None
        return document

    async def aget(
        self, key: str, collection: str = DEFAULT_COLLECTION
    ) -> Optional[dict]:
        """
        Get a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        self._check_async_client()

        # Create collection if it does not exist
        collection = self._sanitize_collection_name(collection)
        self._create_collection_if_not_exists(collection)

        db_collection = self._ascope.collection(collection)
        try:
            return (await db_collection.get(key)).content_as[dict]
        except DocumentNotFoundException:
            return None

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """
        Get all the key-value pairs from the store.

        Args:
            collection (str): collection name

        """
        output = {}

        # Create collection if it does not exist
        collection = self._sanitize_collection_name(collection)
        self._create_collection_if_not_exists(collection)

        db_collection = self._scope.collection(collection)
        results = db_collection.scan(RangeScan())

        for result in results:
            output[result.id] = result.content_as[dict]

        return output

    async def aget_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """
        Get all the key-value pairs from the store.

        Args:
            collection (str): collection name

        """
        self._check_async_client()
        output = {}

        # Create collection if it does not exist
        collection = self._sanitize_collection_name(collection)
        self._create_collection_if_not_exists(collection)

        db_collection = self._ascope.collection(collection)
        results = db_collection.scan(RangeScan())
        async for result in results:
            output[result.id] = result.content_as[dict]

        return output

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """
        Delete a key-value pair from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        # Create collection if it does not exist
        collection = self._sanitize_collection_name(collection)
        self._create_collection_if_not_exists(collection)

        db_collection = self._scope.collection(collection)
        try:
            db_collection.remove(key)
            return True
        except DocumentNotFoundException:
            return False

    async def adelete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """
        Delete a key-value pair from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        self._check_async_client()

        # Create collection if it does not exist
        collection = self._sanitize_collection_name(collection)
        self._create_collection_if_not_exists(collection)

        db_collection = self._ascope.collection(collection)
        try:
            await db_collection.remove(key)
            return True
        except DocumentNotFoundException:
            return False

    @classmethod
    def from_couchbase_client(
        cls,
        client: Cluster,
        bucket_name: str,
        scope_name: str,
        async_client: AsyncCluster = None,
    ) -> "CouchbaseKVStore":
        """
        Initialize a CouchbaseKVStore from a Couchbase cluster object.

        Args:
            cluster (Cluster): Couchbase cluster object
            bucket_name (str): Name of the bucket to use for the key-value store
            scope_name (str): Name of the scope to use for the key-value store

        Returns:
            CouchbaseKVStore: instance of CouchbaseKVStore

        """
        return cls(client, bucket_name, scope_name, async_client)
