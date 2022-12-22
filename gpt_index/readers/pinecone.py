"""Pinecone reader."""

from typing import Any, List

from gpt_index.readers.base import BaseReader
from gpt_index.readers.schema.base import Document


class PineconeReader(BaseReader):
    """Pinecone reader.

    Args:
        api_key (str): Pinecone API key.
        environment (str): Pinecone environment.
    """

    def __init__(self, api_key: str, environment: str):
        """Initialize with parameters."""
        try:
            import pinecone  # noqa: F401
        except ImportError:
            raise ValueError(
                "`pinecone` package not found, please run `pip install pinecone-client`"
            )

        self._api_key = api_key
        self._environment = environment
        pinecone.init(api_key=api_key, environment=environment)

    def load_data(self, **load_kwargs: Any) -> List[Document]:
        """Load data from Pinecone.

        Args:
            index_name (str): Name of the index.
            id_to_text_map (Dict[str, str]): A map from ID's to text.
            separate_documents (Optional[bool]): Whether to return separate
                documents per retrieved entry. Defaults to False.
            vector (List[float]): Query vector.
            top_k (int): Number of results to return.
            **query_kwargs: Keyword arguments to pass to the query.
                Arguments are the exact same as those found in
                Pinecone's reference documentation for the
                query method.

        Returns:
            List[Document]: A list of documents.
        """
        import pinecone

        index_name = load_kwargs.pop("index_name", None)
        if index_name is None:
            raise ValueError("Please provide an index name.")
        id_to_text_map = load_kwargs.pop("id_to_text_map", None)
        if id_to_text_map is None:
            raise ValueError(
                "Please provide an id_to_text_map (a map from ID's to text)."
            )
        vector = load_kwargs.pop("vector", None)
        if vector is None:
            raise ValueError("Please provide a vector.")
        top_k = load_kwargs.pop("top_k", None)
        if top_k is None:
            raise ValueError("Please provide a top_k value.")
        separate_documents = load_kwargs.pop("separate_documents", False)

        query_kwargs = load_kwargs
        index = pinecone.Index(index_name)
        response = index.query(top_k=top_k, vector=vector, **query_kwargs)

        documents = []
        for match in response.matches:
            if match.id not in id_to_text_map:
                raise ValueError("ID not found in id_to_text_map.")
            text = id_to_text_map[match.id]
            documents.append(Document(text=text))

        if not separate_documents:
            text_list = [doc.get_text() for doc in documents]
            text = "\n\n".join(text_list)
            documents = [Document(text=text)]

        return documents
