"""Weaviate reader."""

from typing import Any, List, Optional

from gpt_index.readers.base import BaseReader
from gpt_index.readers.schema.base import Document


class WeaviateReader(BaseReader):
    """Weaviate reader.

    Retrieves documents from Weaviate through vector lookup. Allows option
    to concatenate retrieved documents into one Document, or to return
    separate Document objects per document.

    Args:
        host (str): host.
        auth_client_secret (Optional[weaviate.auth.AuthCredentials]):
            auth_client_secret.
    """

    def __init__(
        self,
        host: str,
        auth_client_secret: Optional[Any] = None,
    ) -> None:
        """Initialize with parameters."""
        try:
            import weaviate  # noqa: F401
            from weaviate import Client  # noqa: F401
            from weaviate.auth import AuthCredentials  # noqa: F401
        except ImportError:
            raise ValueError(
                "`weaviate` package not found, please run `pip install weaviate-client`"
            )

        self.client: Client = Client(host, auth_client_secret=auth_client_secret)

    def load_data(self, **load_kwargs: Any) -> List[Document]:
        """Load data from Weaviate.

        If `graphql_query` is not found in load_kwargs, we assume that
        `class_name` and `properties` are provided.

        Args:
            class_name (Optional[str]): class_name to retrieve documents from.
            properties (Optional[List[str]]): properties to retrieve from documents.
            graphql_query (Optional[str]): Raw GraphQL Query.
                We assume that the query is a Get query.
            separate_documents (Optional[bool]): Whether to return separate
                documents. Defaults to False.

        Returns:
            List[Document]: A list of documents.

        """
        separate_documents = load_kwargs.get("separate_documents", False)
        class_name = None
        if "class_name" in load_kwargs and "properties" in load_kwargs:
            class_name = load_kwargs["class_name"]
            properties = load_kwargs["properties"]
            props_txt = "\n".join(properties)
            graphql_query = f"""
            {{
                Get {{
                    {class_name} {{
                        {props_txt}
                    }}
                }}
            }}
            """
        elif "graphql_query" in load_kwargs:
            graphql_query = load_kwargs["graphql_query"]
        else:
            raise ValueError("`graphql_query` not found in load_kwargs.")

        response = self.client.query.raw(graphql_query)
        if "errors" in response:
            raise ValueError("Invalid query, got errors: {}".format(response["errors"]))

        data_response = response["data"]
        if "Get" not in data_response:
            raise ValueError("Invalid query response, must be a Get query.")

        if class_name is None:
            # infer class_name if only graphql_query was provided
            class_name = list(data_response["Get"].keys())[0]
        entries = data_response["Get"][class_name]
        documents = []
        for entry in entries:
            # for each entry, join properties into <property>:<value>
            # separated by newlines
            text_list = [f"{k}: {v}" for k, v in entry.items()]
            text = "\n".join(text_list)
            documents.append(Document(text=text))

        if not separate_documents:
            # join all documents into one
            text_list = [doc.get_text() for doc in documents]
            text = "\n\n".join(text_list)
            documents = [Document(text=text)]

        return documents
