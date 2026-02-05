"""Zvec reader."""

import json
from typing import Dict, List, Optional
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
import zvec


class ZvecReader(BaseReader):
    """Zvec reader."""

    def __init__(self, path: str):
        """Initialize with parameters."""
        super().__init__()
        try:
            self._collection = zvec.open(path)
        except Exception as e:
            raise ValueError(f"Failed to open zvec collection: {e}")

    def load_data(
        self,
        vector: Optional[List[float]],
        topk: int,
        filter: Optional[str] = None,
        include_vector: bool = True,
        output_fields: Optional[List[str]] = None,
        sparse_vector: Optional[Dict[int, float]] = None,
    ) -> List[Document]:
        """
        Load data from Zvec.

        Args:
            vector (List[float]): Query vector.
            topk (int): Number of results to return.
            filter (Optional[str]): doc fields filter.
            include_vector (bool): Whether to include the embedding in the response.Defaults to True.
            output_fields (Optional[List[str]]): The fields
                to return. Defaults to None, meaning all fields.
            sparse_vector (Optional[Dict[int, float]]): Sparse vector for hybrid search.

        Returns:
            List[Document]: A list of documents.

        """
        vectors = [zvec.VectorQuery(field_name="dense_embedding", vector=vector)]

        if sparse_vector:
            vectors.append(
                zvec.VectorQuery(
                    field_name="sparse_embedding", sparse_vector=sparse_vector
                )
            )

        ret = self._collection.query(
            vectors=vectors,
            topk=topk,
            filter=filter,
            include_vector=include_vector,
            output_fields=output_fields,
        )
        if not ret:
            raise Exception(f"Failed to query document,Error: {ret}")

        documents = []
        for doc in ret:
            meta_dict = json.loads(doc.fields["metadata_"])
            document = Document(
                id_=doc.id,
                text=doc.fields["text"],
                metadata=meta_dict,
                embedding=doc.vectors["dense_embedding"],
            )

            documents.append(document)

        return documents
