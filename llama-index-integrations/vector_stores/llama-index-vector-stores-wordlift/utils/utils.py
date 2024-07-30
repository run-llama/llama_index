from typing import List

import wordlift_client
from wordlift_client import NodeRequest, VectorSearchQueryRequest


class VectorSearchService:
    host: str

    def __init__(self, host="https://api.wordlift.io"):
        self.host = host

    async def update_nodes_collection(self, node_request: List[NodeRequest], key: str):
        async with wordlift_client.ApiClient(
            wordlift_client.Configuration(
                host=self.host,
            )
        ) as api_client:
            api_instance = wordlift_client.VectorSearchNodesApi(api_client)

            return await api_instance.update_nodes_collection(
                node_request=node_request, _headers={"Authorization": "Key " + key}
            )

    async def query_nodes_collection(
        self, vector_search_query_request: VectorSearchQueryRequest, key: str
    ):
        async with wordlift_client.ApiClient(
            wordlift_client.Configuration(
                host=self.host,
            )
        ) as api_client:
            api_instance = wordlift_client.VectorSearchQueriesApi(api_client)

            return await api_instance.create_query(
                vector_search_query_request=vector_search_query_request,
                _headers={"Authorization": "Key " + key},
            )


class WordliftVectorQueryServiceException(Exception):
    initial_exception: Exception
    additional_msg: str

    def __init__(self, exception: Exception, msg: str, *args, **kwargs):
        self.initial_exception = exception
        self.additional_msg = msg
        super().__init__(*args)


class WordliftVectorStoreException(Exception):
    initial_exception: Exception
    additional_msg: str

    def __init__(self, exception: Exception, msg: str, *args, **kwargs):
        self.initial_exception = exception
        self.additional_msg = msg
        super().__init__(*args)
