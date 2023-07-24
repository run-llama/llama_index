"""Marqo reader."""

from typing import Any, Dict, List, Optional

from llama_index.readers.base import BaseReader
from llama_index.schema import Document
from llama_index.vector_stores.utils import DEFAULT_TEXT_KEY
import logging

ID_KEY = "_id"


class MarqoReader(BaseReader):
    """Marqo reader.

    Args:
        api_key (str): Marqo API key.
        url (str): Marqo url.
    """

    def __init__(self, url: str, api_key: Optional[str] = None):
        """Initialize with parameters."""
        # Necessary Marqo initialization steps here.
        try:
            import marqo  # noqa: F401
        except ImportError:
            raise ImportError(
                "`marqo` package not found, please run `pip install marqo`"
            )
        self._api_key = api_key
        self._url = url
        self._text_key = DEFAULT_TEXT_KEY
        self.mq = marqo.Client(api_key=self._api_key, url=self._url)

    def _ensure_index(self, index_name: str) -> None:
        """Ensure the index exists, creating it if necessary."""
        indexes = [index.index_name for index in self.mq.get_indexes()["results"]]
        if index_name not in indexes:
            self.mq.create_index(index_name)
            logging.info(f"Created index {index_name}.")

    def load_data(
        self,
        index_name: str,
        top_k: int,
        separate_documents: bool = True,
        include_vectors: bool = False,
        _text_key: str = DEFAULT_TEXT_KEY,
        searchable_attributes: Optional[Dict[str, str]] = None,
        **query_kwargs: Any,
    ) -> List[Document]:

        """
        Load data from Marqo.

        Args:
            index_name (str): Name of the index.
            top_k (int): Number of results to return.
            separate_documents (bool): Whether to return separate
                documents per retrieved entry. Defaults to True.
            include_vectors (bool): Whether to include vector data in the results.
            _text_key (str): Key to access the main text content of a document.
                Defaults to DEFAULT_TEXT_KEY.
            searchable_attributes (Optional[Dict[str, str]]):
                Optional map from field names to search terms.
            **query_kwargs: Keyword arguments to pass to the query.

        Returns:
            List[Document]: A list of documents.
        """

        self._ensure_index(index_name=index_name)  # Ensure the index exists
        self._text_key = _text_key

        # Construct filter string from searchable_attributes if provided
        filter_string = ""
        if searchable_attributes:
            filter_string = " AND ".join(
                [f"{field}:({term})" for field, term in searchable_attributes.items()]
            )

        # Fetch the documents
        query_kwargs["limit"] = top_k
        if filter_string:
            query_kwargs["filter_string"] = filter_string
        results = self.mq.index(index_name).search(
            q="",
            **query_kwargs,
        )

        # If include_vectors is True, get the document embeddings
        if include_vectors:
            results["hits"] = [
                self.mq.index(index_name).get_document(r["_id"], expose_facets=True)
                for r in results["hits"]
            ]

        documents = []
        for result in results["hits"]:
            doc_id = result["_id"]
            text = result[self._text_key]

            if include_vectors:
                # Get the embedding from '_tensor_facets' for the field
                # specified by self._text_key
                embedding = next(
                    (
                        facet["_embedding"]
                        for facet in result["_tensor_facets"]
                        if facet[_text_key] == text
                    ),
                    None,
                )
            else:
                embedding = None

            documents.append(Document(doc_id=doc_id, text=text, embedding=embedding))

        if not separate_documents:
            text_list = [doc.get_content() for doc in documents]
            text = "\n\n".join(text_list)
            documents = [Document(text=text)]

        return documents
