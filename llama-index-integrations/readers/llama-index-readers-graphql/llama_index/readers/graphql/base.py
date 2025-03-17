"""GraphQL Reader."""

from typing import Dict, List, Optional

import yaml
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class GraphQLReader(BaseReader):
    """
    GraphQL reader.

    Combines all GraphQL results into the Document used by LlamaIndex.

    Args:
        uri (str): GraphQL uri.
        headers (Optional[Dict]): Optional http headers.

    """

    def __init__(
        self,
        uri: Optional[str] = None,
        headers: Optional[Dict] = None,
    ) -> None:
        """Initialize with parameters."""
        try:
            from gql import Client
            from gql.transport.requests import RequestsHTTPTransport

        except ImportError:
            raise ImportError("`gql` package not found, please run `pip install gql`")
        if uri:
            if uri is None:
                raise ValueError("`uri` must be provided.")
            if headers is None:
                headers = {}
            transport = RequestsHTTPTransport(url=uri, headers=headers)
            self.client = Client(transport=transport, fetch_schema_from_transport=True)

    def load_data(self, query: str, variables: Optional[Dict] = None) -> List[Document]:
        """
        Run query with optional variables and turn results into documents.

        Args:
            query (str): GraphQL query string.
            variables (Optional[Dict]): optional query parameters.

        Returns:
            List[Document]: A list of documents.

        """
        try:
            from gql import gql

        except ImportError:
            raise ImportError("`gql` package not found, please run `pip install gql`")
        if variables is None:
            variables = {}

        documents = []

        result = self.client.execute(gql(query), variable_values=variables)

        for key in result:
            entry = result[key]
            if isinstance(entry, list):
                documents.extend([Document(text=yaml.dump(v)) for v in entry])
            else:
                documents.append(Document(text=yaml.dump(entry)))

        return documents
