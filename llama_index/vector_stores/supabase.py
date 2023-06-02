
from typing import Any, List

from vecs.client import Client

from llama_index.vector_stores.types import (NodeWithEmbedding, VectorStore,
                                             VectorStoreQuery,
                                             VectorStoreQueryResult)


class SupabaseVectorStore(VectorStore): 

    
    stores_text = True
    def __init__(self, postgres_connection_string: str, collection_name: str, **kwargs: Any) -> None:
        import_err_msg = "`vecs` package not found, please run `pip install vecs`"
        try:
            import vecs
        except ImportError:
            raise ImportError(import_err_msg)
        
        client = vecs.create_client(postgres_connection_string)

        self._collection = client.get_collection(name = collection_name)

        

    @property
    def client(self) -> None:
        """Get client."""
        return None

    def add(self, embedding_results: List[NodeWithEmbedding]) -> List[str]:
        print("collection:", self._collection)
        # if not self._collection:
        #     raise ValueError("Collection not initialized")
        
        data = []
        ids = []
        for result in embedding_results:
            data.append((result.id, result.embedding, result.node.text))
            ids.append(result.id)
        print("supabase vector store class ...")
        print(data)

        self._collection.upsert(vectors=data)
        return ids
    

