"""
Azure Cognitive Search reader.
A loader that fetches documents from specific index.

"""

from typing import List, Optional

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class AzCognitiveSearchReader(BaseReader):
    """
    General reader for any Azure Cognitive Search index reader.

    Args:
        service_name (str): the name of azure cognitive search service.
        search_key (str): provide azure search access key directly.
        index (str): index name

    """

    def __init__(self, service_name: str, searck_key: str, index: str) -> None:
        """Initialize Azure cognitive search service using the search key."""
        import logging

        logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
        logger.setLevel(logging.WARNING)

        azure_credential = AzureKeyCredential(searck_key)

        self.search_client = SearchClient(
            endpoint=f"https://{service_name}.search.windows.net",
            index_name=index,
            credential=azure_credential,
        )

    def load_data(
        self, query: str, content_field: str, filter: Optional[str] = None
    ) -> List[Document]:
        """
        Read data from azure cognitive search index.

        Args:
            query (str): search term in Azure Search index
            content_field (str): field name of the document content.
            filter (str): Filter expression. For example : 'sourcepage eq
                'employee_handbook-3.pdf' and sourcefile eq 'employee_handbook.pdf''

        Returns:
            List[Document]: A list of documents.

        """
        search_result = self.search_client.search(query, filter=filter)

        return [
            Document(
                text=result[content_field],
                extra_info={"id": result["id"], "score": result["@search.score"]},
            )
            for result in search_result
        ]
