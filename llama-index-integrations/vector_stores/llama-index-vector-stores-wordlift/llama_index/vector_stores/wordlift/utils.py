from typing import List

import manager_client
from manager_client import NodeRequest, VectorSearchQueryRequest


class VectorSearchService:
    host: str

    def __init__(self, host='https://api.wordlift.io'):
        self.host = host

    async def update_nodes_collection(self, node_request: List[NodeRequest], key: str):
        async with manager_client.ApiClient(manager_client.Configuration(
                host=self.host,
        )) as api_client:
            api_instance = manager_client.VectorSearchNodesApi(api_client)

            return await api_instance.update_nodes_collection(
                node_request=node_request,
                _headers={'Authorization': 'Key ' + key}
            )

    async def query_nodes_collection(self, vector_search_query_request: VectorSearchQueryRequest, key: str):
        async with manager_client.ApiClient(manager_client.Configuration(
                host=self.host,
        )) as api_client:
            api_instance = manager_client.VectorSearchQueriesApi(api_client)

            return await api_instance.create_query(
                vector_search_query_request=vector_search_query_request,
                _headers={'Authorization': 'Key ' + key}
            )