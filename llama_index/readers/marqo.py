"""Marqo reader."""

from typing import Any, Dict, List, Optional

from llama_index.readers.base import BaseReader
from llama_index.schema import Document
from llama_index.vector_stores.utils import (
    DEFAULT_TEXT_KEY
)
import logging

ID_KEY = "_id"

class MarqoReader(BaseReader):
    """Marqo reader.

    Args:
        api_key (str): Marqo API key.
        url (str): Marqo url.
    """

    def __init__(self, api_key: str, url: str):
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
        self.mq = marqo.Client(url=self._url, api_key=self._api_key)
        

    def _ensure_index(self, index_name: str = None):
        """Ensure the index exists, creating it if necessary."""
        indexes = [index.index_name for index in self._marqo_client.get_indexes()["results"]]
        if index_name not in indexes:
            self.mq.create_index(index_name)
            logging.info(f"Created index {index_name}.")
    
    def load_data(
        self,
        index_name: str,
        id_to_text_map: Dict[str, str],
        top_k: int,
        separate_documents: bool = True,
        include_vectors: bool = False,
        _text_key: str = DEFAULT_TEXT_KEY,
        searchable_attributes: Optional[List[str]] = None,
        **query_kwargs: Any
    ) -> List[Document]:
        """
        Load data from Marqo.

        Args:
            index_name (str): Name of the index.
            id_to_text_map (Dict[str, str]): A map from ID's to text.
            top_k (int): Number of results to return.
            separate_documents (bool): Whether to return separate documents per retrieved entry. Defaults to True.
            include_vectors (bool): Whether to include vector data in the results.
            searchable_attributes (Optional[List[str]]): The attributes to retrieve from the Marqo index.
            **query_kwargs: Keyword arguments to pass to the query.

        Returns:
            List[Document]: A list of documents.
        """
        import marqo 
        self._ensure_index(index_name=index_name)  # Ensure the index exists
        self._text_key = _text_key

        # Construct filter string from searchable_attributes
        filter_string = None
        if searchable_attributes:
            filter_string = " AND ".join([f"{attr}:{id_to_text_map[id]}" for attr in searchable_attributes for id in id_to_text_map.keys()])

        # Fetch the documents by their IDs
        results = self.mq.index(index_name).search(
            q="",
            limit=top_k,
            filter_string=filter_string,
            **query_kwargs,
        )

        print("\n\results:", results)
        # If include_vectors is True, get the document embeddings
        if include_vectors:
            results["hits"] = [self.mq.index(index_name).get_document(r["_id"], expose_facets=True) for r in results["hits"]]

        documents = []
        for result in results["hits"]:
            doc_id = result["_id"]
            text = result[self._text_key]
            assert text == id_to_text_map[doc_id]

            if include_vectors:
                # Get the embedding from '_tensor_facets' for the field specified by self._text_key
                embedding = next((facet['_embedding'] for facet in result['_tensor_facets'] if facet[_text_key] == text), None)
            else:
                embedding = None

            documents.append(Document(doc_id=doc_id,text=text, embedding=embedding))

        if not separate_documents:
            text_list = [doc.get_content() for doc in documents]
            text = "\n\n".join(text_list)
            documents = [Document(text=text)]

        print("\n\ndocuments:", documents)
        return documents
