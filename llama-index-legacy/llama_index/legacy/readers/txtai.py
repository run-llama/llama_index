"""txtai reader."""

from typing import Any, Dict, List

import numpy as np

from llama_index.legacy.readers.base import BaseReader
from llama_index.legacy.schema import Document


class TxtaiReader(BaseReader):
    """txtai reader.

    Retrieves documents through an existing in-memory txtai index.
    These documents can then be used in a downstream LlamaIndex data structure.
    If you wish use txtai itself as an index to to organize documents,
    insert documents, and perform queries on them, please use VectorStoreIndex
    with TxtaiVectorStore.

    Args:
        txtai_index (txtai.ann.ANN): A txtai Index object (required)

    """

    def __init__(self, index: Any):
        """Initialize with parameters."""
        import_err_msg = """
            `txtai` package not found. For instructions on
            how to install `txtai` please visit
            https://neuml.github.io/txtai/install/
        """
        try:
            import txtai  # noqa
        except ImportError:
            raise ImportError(import_err_msg)

        self._index = index

    def load_data(
        self,
        query: np.ndarray,
        id_to_text_map: Dict[str, str],
        k: int = 4,
        separate_documents: bool = True,
    ) -> List[Document]:
        """Load data from txtai index.

        Args:
            query (np.ndarray): A 2D numpy array of query vectors.
            id_to_text_map (Dict[str, str]): A map from ID's to text.
            k (int): Number of nearest neighbors to retrieve. Defaults to 4.
            separate_documents (Optional[bool]): Whether to return separate
                documents. Defaults to True.

        Returns:
            List[Document]: A list of documents.

        """
        search_result = self._index.search(query, k)
        documents = []
        for query_result in search_result:
            for doc_id, _ in query_result:
                doc_id = str(doc_id)
                if doc_id not in id_to_text_map:
                    raise ValueError(
                        f"Document ID {doc_id} not found in id_to_text_map."
                    )
                text = id_to_text_map[doc_id]
                documents.append(Document(text=text))

        if not separate_documents:
            # join all documents into one
            text_list = [doc.get_content() for doc in documents]
            text = "\n\n".join(text_list)
            documents = [Document(text=text)]

        return documents
