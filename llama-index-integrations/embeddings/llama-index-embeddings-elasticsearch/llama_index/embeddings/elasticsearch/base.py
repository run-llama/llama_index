import asyncio
from typing import Any, List

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr

from elasticsearch import Elasticsearch
from elasticsearch.client import MlClient


class ElasticsearchEmbedding(BaseEmbedding):
    """
    Elasticsearch embedding models.

    This class provides an interface to generate embeddings using a model deployed
    in an Elasticsearch cluster. It requires an Elasticsearch connection object
    and the model_id of the model deployed in the cluster.

    In Elasticsearch you need to have an embedding model loaded and deployed.
    - https://www.elastic.co
        /guide/en/elasticsearch/reference/current/infer-trained-model.html
    - https://www.elastic.co
        /guide/en/machine-learning/current/ml-nlp-deploy-models.html
    """  #

    _client: Any = PrivateAttr()
    model_id: str
    input_field: str

    def class_name(self) -> str:
        return "ElasticsearchEmbedding"

    def __init__(
        self,
        client: Any,
        model_id: str,
        input_field: str = "text_field",
        **kwargs: Any,
    ):
        super().__init__(model_id=model_id, input_field=input_field, **kwargs)
        self._client = client

    @classmethod
    def from_es_connection(
        cls,
        model_id: str,
        es_connection: Any,
        input_field: str = "text_field",
    ) -> BaseEmbedding:
        """
        Instantiate embeddings from an existing Elasticsearch connection.

        This method provides a way to create an instance of the ElasticsearchEmbedding
        class using an existing Elasticsearch connection. The connection object is used
        to create an MlClient, which is then used to initialize the
        ElasticsearchEmbedding instance.

        Args:
        model_id (str): The model_id of the model deployed in the Elasticsearch cluster.
        es_connection (elasticsearch.Elasticsearch): An existing Elasticsearch
            connection object.
        input_field (str, optional): The name of the key for the input text field
            in the document. Defaults to 'text_field'.

        Returns:
        ElasticsearchEmbedding: An instance of the ElasticsearchEmbedding class.

        Example:
            .. code-block:: python

                from elasticsearch import Elasticsearch

                from llama_index.embeddings.elasticsearch import ElasticsearchEmbedding

                # Define the model ID and input field name (if different from default)
                model_id = "your_model_id"
                # Optional, only if different from 'text_field'
                input_field = "your_input_field"

                # Create Elasticsearch connection
                es_connection = Elasticsearch(hosts=["localhost:9200"], basic_auth=("user", "password"))

                # Instantiate ElasticsearchEmbedding using the existing connection
                embeddings = ElasticsearchEmbedding.from_es_connection(
                    model_id,
                    es_connection,
                    input_field=input_field,
                )

        """
        client = MlClient(es_connection)
        return cls(client, model_id, input_field=input_field)

    @classmethod
    def from_credentials(
        cls,
        model_id: str,
        es_url: str,
        es_username: str,
        es_password: str,
        input_field: str = "text_field",
    ) -> BaseEmbedding:
        """
        Instantiate embeddings from Elasticsearch credentials.

        Args:
            model_id (str): The model_id of the model deployed in the Elasticsearch
                cluster.
            input_field (str): The name of the key for the input text field in the
                document. Defaults to 'text_field'.
            es_url: (str): The Elasticsearch url to connect to.
            es_username: (str): Elasticsearch username.
            es_password: (str): Elasticsearch password.

        Example:
            .. code-block:: python

                from llama_index.embeddings.bedrock import ElasticsearchEmbedding

                # Define the model ID and input field name (if different from default)
                model_id = "your_model_id"
                # Optional, only if different from 'text_field'
                input_field = "your_input_field"

                embeddings = ElasticsearchEmbedding.from_credentials(
                    model_id,
                    input_field=input_field,
                    es_url="foo",
                    es_username="bar",
                    es_password="baz",
                )

        """
        es_connection = Elasticsearch(
            hosts=[es_url],
            basic_auth=(es_username, es_password),
        )

        client = MlClient(es_connection)
        return cls(client, model_id, input_field=input_field)

    def _get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a single query text.

        Args:
            text (str): The query text to generate an embedding for.

        Returns:
            List[float]: The embedding for the input query text.

        """
        response = self._client.infer_trained_model(
            model_id=self.model_id,
            docs=[{self.input_field: text}],
        )

        return response["inference_results"][0]["predicted_value"]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await asyncio.to_thread(self._get_text_embedding, text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)


ElasticsearchEmbeddings = ElasticsearchEmbedding
