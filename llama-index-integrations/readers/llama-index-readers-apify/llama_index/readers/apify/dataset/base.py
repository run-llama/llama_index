"""Apify dataset reader."""

from typing import Callable, Dict, List

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class ApifyDataset(BaseReader):
    """
    Apify Dataset reader.
    Reads a dataset on the Apify platform.

    Args:
        apify_api_token (str): Apify API token.

    """

    def __init__(self, apify_api_token: str) -> None:
        """Initialize Apify dataset reader."""
        from apify_client import ApifyClient

        client = ApifyClient(apify_api_token)
        if hasattr(client.http_client, "httpx_client"):
            client.http_client.httpx_client.headers["user-agent"] += (
                "; Origin/llama_index"
            )

        self.apify_client = client

    def load_data(
        self, dataset_id: str, dataset_mapping_function: Callable[[Dict], Document]
    ) -> List[Document]:
        """
        Load data from the Apify dataset.

        Args:
            dataset_id (str): Dataset ID.
            dataset_mapping_function (Callable[[Dict], Document]): Function to map dataset items to Document.


        Returns:
            List[Document]: List of documents.

        """
        items_list = self.apify_client.dataset(dataset_id).list_items(clean=True)

        document_list = []
        for item in items_list.items:
            document = dataset_mapping_function(item)
            if not isinstance(document, Document):
                raise ValueError("Dataset_mapping_function must return a Document")
            document_list.append(document)

        return document_list
