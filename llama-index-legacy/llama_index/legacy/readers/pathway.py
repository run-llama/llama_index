"""Pathway reader."""

from typing import List, Optional, Union

from llama_index.legacy.readers.base import BaseReader
from llama_index.legacy.schema import Document


class PathwayReader(BaseReader):
    """Pathway reader.

    Retrieve documents from Pathway data indexing pipeline.

    Args:
        host (str): The URI where Pathway is currently hosted.
        port (str | int): The port number on which Pathway is listening.

    See Also:
        llamaindex.retriever.pathway.PathwayRetriever and,
        llamaindex.retriever.pathway.PathwayVectorServer
    """

    def __init__(self, host: str, port: Union[str, int]):
        """Initializing the Pathway reader client."""
        import_err_msg = "`pathway` package not found, please run `pip install pathway`"
        try:
            from pathway.xpacks.llm.vector_store import VectorStoreClient
        except ImportError:
            raise ImportError(import_err_msg)
        self.client = VectorStoreClient(host, port)

    def load_data(
        self,
        query_text: str,
        k: Optional[int] = 4,
        metadata_filter: Optional[str] = None,
    ) -> List[Document]:
        """Load data from Pathway.

        Args:
            query_text (str): The text to get the closest neighbors of.
            k (int): Number of results to return.
            metadata_filter (str): Filter to be applied.

        Returns:
            List[Document]: A list of documents.
        """
        results = self.client(query_text, k, metadata_filter)
        documents = []
        for return_elem in results:
            document = Document(
                text=return_elem["text"],
                extra_info=return_elem["metadata"],
            )

            documents.append(document)

        return documents
