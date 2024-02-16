"""Awadb reader."""

from typing import Any, List

import numpy as np

from llama_index.legacy.readers.base import BaseReader
from llama_index.legacy.schema import Document


class AwadbReader(BaseReader):
    """Awadb reader.

    Retrieves documents through an existing awadb client.
    These documents can then be used in a downstream LlamaIndex data structure.

    Args:
        client (awadb.client): An awadb client.

    """

    def __init__(self, client: Any):
        """Initialize with parameters."""
        import_err_msg = """
            `faiss` package not found. For instructions on
            how to install `faiss` please visit
            https://github.com/facebookresearch/faiss/wiki/Installing-Faiss
        """
        try:
            pass
        except ImportError:
            raise ImportError(import_err_msg)

        self.awadb_client = client

    def load_data(
        self,
        query: np.ndarray,
        k: int = 4,
        separate_documents: bool = True,
    ) -> List[Document]:
        """Load data from Faiss.

        Args:
            query (np.ndarray): A 2D numpy array of query vectors.
            k (int): Number of nearest neighbors to retrieve. Defaults to 4.
            separate_documents (Optional[bool]): Whether to return separate
                documents. Defaults to True.

        Returns:
            List[Document]: A list of documents.

        """
        results = self.awadb_client.Search(
            query,
            k,
            text_in_page_content=None,
            meta_filter=None,
            not_include_fields=None,
        )
        documents = []
        for item_detail in results[0]["ResultItems"]:
            documents.append(Document(text=item_detail["embedding_text"]))

        if not separate_documents:
            # join all documents into one
            text_list = [doc.get_content() for doc in documents]
            text = "\n\n".join(text_list)
            documents = [Document(text=text)]

        return documents
