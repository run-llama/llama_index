"""Weaviate reader."""

from typing import Any, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class WeaviateReader(BaseReader):
    """
    Weaviate reader.

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
            import weaviate  # noqa
            from weaviate import Client
            from weaviate.auth import AuthCredentials  # noqa
        except ImportError:
            raise ImportError(
                "`weaviate` package not found, please run `pip install weaviate-client`"
            )

        self.client: Client = Client(host, auth_client_secret=auth_client_secret)

    def load_data(
        self,
        class_name: Optional[str] = None,
        properties: Optional[List[str]] = None,
        graphql_query: Optional[str] = None,
        separate_documents: Optional[bool] = True,
    ) -> List[Document]:
        """
        Load data from Weaviate.

        If `graphql_query` is not found in load_kwargs, we assume that
        `class_name` and `properties` are provided.

        Args:
            class_name (Optional[str]): class_name to retrieve documents from.
            properties (Optional[List[str]]): properties to retrieve from documents.
            graphql_query (Optional[str]): Raw GraphQL Query.
                We assume that the query is a Get query.
            separate_documents (Optional[bool]): Whether to return separate
                documents. Defaults to True.

        Returns:
            List[Document]: A list of documents.

        """
        if class_name is not None and properties is not None:
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
        elif graphql_query is not None:
            pass
        else:
            raise ValueError(
                "Either `class_name` and `properties` must be specified, "
                "or `graphql_query` must be specified."
            )

        response = self.client.query.raw(graphql_query)
        if "errors" in response:
            raise ValueError("Invalid query, got errors: {}".format(response["errors"]))

        data_response = response["data"]
        if "Get" not in data_response:
            raise ValueError("Invalid query response, must be a Get query.")

        if class_name is None:
            # infer class_name if only graphql_query was provided
            class_name = next(iter(data_response["Get"].keys()))
        entries = data_response["Get"][class_name]
        documents = []
        for entry in entries:
            embedding: Optional[List[float]] = None
            # for each entry, join properties into <property>:<value>
            # separated by newlines
            text_list = []
            for k, v in entry.items():
                if k == "_additional":
                    if "vector" in v:
                        embedding = v["vector"]
                    continue
                text_list.append(f"{k}: {v}")

            text = "\n".join(text_list)
            documents.append(Document(text=text, embedding=embedding))

        if not separate_documents:
            # join all documents into one
            text_list = [doc.get_content() for doc in documents]
            text = "\n\n".join(text_list)
            documents = [Document(text=text)]

        return documents
