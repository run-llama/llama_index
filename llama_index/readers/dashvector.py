"""DashVector reader."""

from typing import Dict, List, Optional

from llama_index.readers.base import BaseReader
from llama_index.schema import Document


class DashVectorReader(BaseReader):
    """DashVector reader.

    Args:
        api_key (str): DashVector API key.
    """

    def __init__(self, api_key: str):
        """Initialize with parameters."""
        try:
            import dashvector
        except ImportError:
            raise ImportError(
                "`dashvector` package not found, please run `pip install dashvector`"
            )

        self._api_key = api_key
        self._client = dashvector.Client(api_key=api_key)

    def load_data(
        self,
        collection_name: str,
        id_to_text_map: Dict[str, str],
        vector: Optional[List[float]],
        top_k: int,
        separate_documents: bool = True,
        filter: Optional[str] = None,
        include_vector: bool = True,
    ) -> List[Document]:
        """Load data from DashVector.

        Args:
            collection_name (str): Name of the collection.
            id_to_text_map (Dict[str, str]): A map from ID's to text.
            separate_documents (Optional[bool]): Whether to return separate
                documents per retrieved entry. Defaults to True.
            vector (List[float]): Query vector.
            top_k (int): Number of results to return.
            filter (Optional[str]): doc fields filter conditions that meet the SQL
                where clause specification.
            include_vector (bool): Whether to include the embedding in the response.
                Defaults to True.

        Returns:
            List[Document]: A list of documents.
        """
        collection = self._client.get(collection_name)
        if not collection:
            raise ValueError(
                f"Failed to get collection: {collection_name}," f"Error: {collection}"
            )

        resp = collection.query(
            vector=vector,
            topk=top_k,
            filter=filter,
            include_vector=include_vector,
        )
        if not resp:
            raise Exception(f"Failed to query document," f"Error: {resp}")

        documents = []
        for doc in resp:
            if doc.id not in id_to_text_map:
                raise ValueError("ID not found in id_to_text_map.")
            text = id_to_text_map[doc.id]
            embedding = doc.vector
            if len(embedding) == 0:
                embedding = None
            documents.append(Document(text=text, embedding=embedding))

        if not separate_documents:
            text_list = [doc.get_content() for doc in documents]
            text = "\n\n".join(text_list)
            documents = [Document(text=text)]

        return documents
