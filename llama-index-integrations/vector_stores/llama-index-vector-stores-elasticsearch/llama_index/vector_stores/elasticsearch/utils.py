from typing import Any, Dict, Optional

from elasticsearch import AsyncElasticsearch, Elasticsearch


def get_user_agent() -> str:
    """Get user agent for Elasticsearch client."""
    import llama_index.core

    version = getattr(llama_index.core, "__version__", "")
    return f"llama_index-py-vs/{version}"


def get_elasticsearch_client(
    url: Optional[str] = None,
    cloud_id: Optional[str] = None,
    api_key: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> AsyncElasticsearch:
    if url and cloud_id:
        raise ValueError(
            "Both es_url and cloud_id are defined. Please provide only one."
        )

    connection_params: Dict[str, Any] = {}

    if url:
        connection_params["hosts"] = [url]
    elif cloud_id:
        connection_params["cloud_id"] = cloud_id
    else:
        raise ValueError("Please provide either elasticsearch_url or cloud_id.")

    if api_key:
        connection_params["api_key"] = api_key
    elif username and password:
        connection_params["basic_auth"] = (username, password)

    sync_es_client = Elasticsearch(
        **connection_params, headers={"user-agent": get_user_agent()}
    )
    async_es_client = AsyncElasticsearch(
        **connection_params, headers={"user-agent": get_user_agent()}
    )

    sync_es_client.info()  # use sync client so don't have to 'await' to just get info

    return async_es_client
