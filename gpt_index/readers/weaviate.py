"""Weaviate reader."""

from typing import Any, Dict, List, Optional

from gpt_index.readers.base import BaseReader
from gpt_index.schema import Document


class WeaviateReader(BaseReader):
    """Weaviate reader.

    Retrieves documents from Weaviate through vector lookup. Allows option
    to concatenate retrieved documents into one Document, or to return 
    separate Document objects per document.

    Args:
        host (str): host.

    """

    def __init__(
        self, 
        host: str,
        auth_client_secret: Optional[Any] = None,
    ) -> None:
        """Initialize with parameters."""
        try:
            import weaviate  # noqa: F401
            from weaviate import Client, # noqa: F401
            from weaviate.auth import AuthCredentials
        except ImportError:
            raise ValueError(
                "`weaviate` package not found, please run `pip install weaviate-client`"
            )
            
        self.client: Client = Client(host, auth_client_secret=auth_client_secret)

    def load_data(self, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory.

        Args:
            graphql_query (str): Raw GraphQL Query.

        Returns:
            List[Document]: A list of documents.

        """
        if "graphql_query" not in load_kwargs:
            raise ValueError("`graphql_query` not found in load_kwargs.")
        else:
            graphql_query = load_kwargs["graphql_query"]

        response = self.client.query.raw(graphql_query)
        print(response)
        raise Exception
        return result

