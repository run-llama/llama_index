from typing import Any, Dict, List, Optional, Tuple
from logging import getLogger

from llama_index.core.storage.kvstore.types import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION,
    BaseKVStore,
)
import asyncio
import nest_asyncio
import elasticsearch
from elasticsearch.helpers import async_bulk, async_scan


logger = getLogger(__name__)

IMPORT_ERROR_MSG = (
    "`elasticsearch` package not found, please run `pip install elasticsearch`"
)


def _get_elasticsearch_client(
    *,
    es_url: Optional[str] = None,
    cloud_id: Optional[str] = None,
    api_key: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> elasticsearch.AsyncElasticsearch:
    """
    Get AsyncElasticsearch client.

    Args:
        es_url: Elasticsearch URL.
        cloud_id: Elasticsearch cloud ID.
        api_key: Elasticsearch API key.
        username: Elasticsearch username.
        password: Elasticsearch password.

    Returns:
        AsyncElasticsearch client.

    Raises:
        ConnectionError: If Elasticsearch client cannot connect to Elasticsearch.

    """
    if es_url and cloud_id:
        raise ValueError(
            "Both es_url and cloud_id are defined. Please provide only one."
        )

    connection_params: Dict[str, Any] = {}

    if es_url:
        connection_params["hosts"] = [es_url]
    elif cloud_id:
        connection_params["cloud_id"] = cloud_id
    else:
        raise ValueError("Please provide either elasticsearch_url or cloud_id.")

    if api_key:
        connection_params["api_key"] = api_key
    elif username and password:
        connection_params["basic_auth"] = (username, password)

    sync_es_client = elasticsearch.Elasticsearch(
        **connection_params,
        headers={"user-agent": ElasticsearchKVStore.get_user_agent()},
    )
    async_es_client = elasticsearch.AsyncElasticsearch(**connection_params)
    try:
        sync_es_client.info()  # so don't have to 'await' to just get info
    except Exception as e:
        logger.error(f"Error connecting to Elasticsearch: {e}")
        raise

    return async_es_client


class ElasticsearchKVStore(BaseKVStore):
    """
    Elasticsearch Key-Value store.

    Args:
        index_name: Name of the Elasticsearch index.
        es_client: Optional. Pre-existing AsyncElasticsearch client.
        es_url: Optional. Elasticsearch URL.
        es_cloud_id: Optional. Elasticsearch cloud ID.
        es_api_key: Optional. Elasticsearch API key.
        es_user: Optional. Elasticsearch username.
        es_password: Optional. Elasticsearch password.


    Raises:
        ConnectionError: If AsyncElasticsearch client cannot connect to Elasticsearch.
        ValueError: If neither es_client nor es_url nor es_cloud_id is provided.

    """

    es_client: Optional[Any]
    es_url: Optional[str]
    es_cloud_id: Optional[str]
    es_api_key: Optional[str]
    es_user: Optional[str]
    es_password: Optional[str]

    def __init__(
        self,
        index_name: str,
        es_client: Optional[Any],
        es_url: Optional[str] = None,
        es_cloud_id: Optional[str] = None,
        es_api_key: Optional[str] = None,
        es_user: Optional[str] = None,
        es_password: Optional[str] = None,
    ) -> None:
        nest_asyncio.apply()

        """Init a ElasticsearchKVStore."""
        try:
            from elasticsearch import AsyncElasticsearch
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)

        if es_client is not None:
            self._client = es_client.options(
                headers={"user-agent": self.get_user_agent()}
            )
        elif es_url is not None or es_cloud_id is not None:
            self._client: AsyncElasticsearch = _get_elasticsearch_client(
                es_url=es_url,
                username=es_user,
                password=es_password,
                cloud_id=es_cloud_id,
                api_key=es_api_key,
            )
        else:
            raise ValueError(
                """Either provide a pre-existing AsyncElasticsearch or valid \
                credentials for creating a new connection."""
            )

    @property
    def client(self) -> Any:
        """Get async elasticsearch client."""
        return self._client

    @staticmethod
    def get_user_agent() -> str:
        """Get user agent for elasticsearch client."""
        return "llama_index-py-vs"

    async def _create_index_if_not_exists(self, index_name: str) -> None:
        """
        Create the AsyncElasticsearch index if it doesn't already exist.

        Args:
            index_name: Name of the AsyncElasticsearch index to create.

        """
        if await self.client.indices.exists(index=index_name):
            logger.debug(f"Index {index_name} already exists. Skipping creation.")

        else:
            index_settings = {"mappings": {"_source": {"enabled": True}}}

            logger.debug(
                f"Creating index {index_name} with mappings {index_settings['mappings']}"
            )
            await self.client.indices.create(index=index_name, **index_settings)

    def put(
        self,
        key: str,
        val: dict,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        """
        Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name

        """
        self.put_all([(key, val)], collection=collection)

    async def aput(
        self,
        key: str,
        val: dict,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        """
        Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name

        """
        await self.aput_all([(key, val)], collection=collection)

    def put_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        return asyncio.get_event_loop().run_until_complete(
            self.aput_all(kv_pairs, collection, batch_size)
        )

    async def aput_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        await self._create_index_if_not_exists(collection)

        # Prepare documents with '_id' set to the key for batch insertion
        docs = [{"_id": key, **value} for key, value in kv_pairs]

        # Insert documents in batches
        for batch in (
            docs[i : i + batch_size] for i in range(0, len(docs), batch_size)
        ):
            requests = []
            for doc in batch:
                doc_id = doc["_id"]
                doc.pop("_id")
                logger.debug(doc)
                request = {
                    "_op_type": "index",
                    "_index": collection,
                    **doc,
                    "_id": doc_id,
                }
                requests.append(request)
            await async_bulk(self.client, requests, chunk_size=batch_size, refresh=True)

    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        """
        Get a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        return asyncio.get_event_loop().run_until_complete(self.aget(key, collection))

    async def aget(
        self, key: str, collection: str = DEFAULT_COLLECTION
    ) -> Optional[dict]:
        """
        Get a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        await self._create_index_if_not_exists(collection)

        try:
            response = await self._client.get(index=collection, id=key, source=True)
            return response.body["_source"]
        except elasticsearch.NotFoundError:
            return None

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """
        Get all values from the store.

        Args:
            collection (str): collection name

        """
        return asyncio.get_event_loop().run_until_complete(self.aget_all(collection))

    async def aget_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """
        Get all values from the store.

        Args:
            collection (str): collection name

        """
        await self._create_index_if_not_exists(collection)

        result = {}
        q = {"query": {"match_all": {}}}
        async for doc in async_scan(client=self._client, index=collection, query=q):
            doc_id = doc["_id"]
            content = doc["_source"]
            result[doc_id] = content
        return result

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """
        Delete a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        return asyncio.get_event_loop().run_until_complete(
            self.adelete(key, collection)
        )

    async def adelete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """
        Delete a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        await self._create_index_if_not_exists(collection)

        try:
            response = await self._client.delete(index=collection, id=key)
            return response.body["result"] == "deleted"
        except elasticsearch.NotFoundError:
            return False
