from typing import Any, Dict, List, Optional, Tuple, cast
from logging import getLogger

from llama_index.core.storage.kvstore.types import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION,
    BaseKVStore,
)

import elasticsearch

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
)-> Any:
    """Get AsyncElasticsearch client.

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
        **connection_params, headers={"user-agent": ElasticsearchKVStore.get_user_agent()}
    )
    async_es_client = elasticsearch.AsyncElasticsearch(**connection_params)
    try:
        sync_es_client.info()  # so don't have to 'await' to just get info
    except Exception as e:
        logger.error(f"Error connecting to Elasticsearch: {e}")
        raise

    return async_es_client

class ElasticsearchKVStore(BaseKVStore):
    """Elasticsearch Key-Value store.

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
        """Init a ElasticsearchKVStore."""
        try:
            from elasticsearch import Elasticsearch
        except ImportError:
            raise ImportError(IMPORT_ERROR_MSG)
        
        if es_client is not None:
            self._client = es_client.options(
                headers={"user-agent": self.get_user_agent()}
            )
        elif es_url is not None or es_cloud_id is not None:
            self._client = _get_elasticsearch_client(
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

        self._index_name = index_name or "db_docstore"

