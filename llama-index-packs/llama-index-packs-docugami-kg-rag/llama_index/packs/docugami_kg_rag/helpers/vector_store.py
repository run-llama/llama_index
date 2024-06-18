from typing import Optional

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.packs.docugami_kg_rag.config import CHROMA_DIRECTORY, EMBEDDINGS

import chromadb


def get_vector_store(docset_id) -> Optional[ChromaVectorStore]:
    db = chromadb.PersistentClient(path=str(CHROMA_DIRECTORY.absolute()))
    chroma_collection = db.get_or_create_collection(docset_id)
    return ChromaVectorStore(
        chroma_collection=chroma_collection, embed_model=EMBEDDINGS
    )


def get_vector_store_index(docset_id, embedding) -> Optional[VectorStoreIndex]:
    vector_store = get_vector_store(docset_id)
    return VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embedding,
    )
