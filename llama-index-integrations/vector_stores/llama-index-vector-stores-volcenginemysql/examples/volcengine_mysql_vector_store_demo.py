from __future__ import annotations

from typing import List
import os
from volcenginesdkarkruntime import Ark

from llama_index.core import Document, StorageContext, VectorStoreIndex, Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.vector_stores.types import VectorStoreQuery

from llama_index.vector_stores.volcengine_mysql import VolcengineMySQLVectorStore

EMBED_DIM: int = 2048


# Required environment variables for this demo:
# - ARK_API_KEY: API key for Volcengine Ark.
# - ARK_EMBEDDING_MODEL: embedding endpoint/model ID used by ArkEmbedding.
# - ARK_LLM_MODEL: chat completion model endpoint ID used by call_ark_llm.
# - VEM_HOST, VEM_PORT, VEM_USER, VEM_PASSWORD, VEM_DATABASE: MySQL
#   connection info for VolcengineMySQLVectorStore.
# - VEM_TABLE (optional): MySQL table name for the vector store; defaults to
#   "llamaindex_vs_local_dummy_demo" if not set.

EMBEDDING_MODEL = os.getenv("ARK_EMBEDDING_MODEL")
if not EMBEDDING_MODEL:
    raise RuntimeError("Please set ARK_EMBEDDING_MODEL environment variable.")

LLM_MODEL = os.getenv("ARK_LLM_MODEL")
if not LLM_MODEL:
    raise RuntimeError("Please set ARK_LLM_MODEL environment variable.")

ARK_API_KEY = os.getenv("ARK_API_KEY")
if not ARK_API_KEY:
    raise RuntimeError("Please set ARK_API_KEY environment variable for Ark.")

"""Ark client configured identically to the working sample in ~/dev/test/ark.py.

By relying on the SDK defaults (no custom base_url) we avoid proxy issues that
occur when overriding the API host, and we let the SDK route requests correctly
for both embedding and chat endpoints.
"""
ARK_CLIENT = Ark(api_key=ARK_API_KEY)


class ArkEmbedding(BaseEmbedding):
    """Embedding model backed by Volcengine Ark embeddings."""

    def __init__(self, client: Ark, model: str = EMBEDDING_MODEL) -> None:
        super().__init__()
        self._client = client
        self._model = model

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:  # type: ignore[override]
        if not texts:
            return []

        vectors: List[List[float]] = []
        for text in texts:
            try:
                # Match the working multimodal embeddings usage from ~/dev/test/ark.py
                # while still only sending text. Each input is wrapped as a
                # multimodal "text" block to conform to the endpoint.
                resp = self._client.multimodal_embeddings.create(
                    model=self._model,
                    input=[{"type": "text", "text": text}],
                )
                vectors.append(resp.data.embedding)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(f"Ark embeddings request failed: {exc}") from exc

        return vectors

    def _get_text_embedding(self, text: str) -> List[float]:  # type: ignore[override]
        return self._get_text_embeddings([text])[0]

    def _get_query_embedding(self, query: str) -> List[float]:  # type: ignore[override]
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:  # type: ignore[override]
        return self._get_query_embedding(query)


def call_ark_llm(prompt: str) -> str:
    """Call Ark chat completion using the same LLM endpoint."""
    completion = ARK_CLIENT.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        reasoning_effort="medium",
        extra_headers={"x-is-encrypted": "true"},
    )
    return completion.choices[0].message.content


def build_vector_store() -> VolcengineMySQLVectorStore:
    """Initialize VolcengineMySQLVectorStore with local connection params.

    Uses the same instance as the user's other demos, but this function
    can be adapted easily to different hosts/credentials.
    """

    # MySQL connection parameters are read from environment variables and must
    # be provided by the user for this demo.
    host = os.getenv("VEM_HOST")
    port = int(os.getenv("VEM_PORT")) if os.getenv("VEM_PORT") else None
    user = os.getenv("VEM_USER")
    password = os.getenv("VEM_PASSWORD")
    database = os.getenv("VEM_DATABASE")
    table_name = os.getenv("VEM_TABLE", "llamaindex_demo")

    if not all([host, port, user, password, database]):
        raise RuntimeError(
            "Please set VEM_HOST, VEM_PORT, VEM_USER, VEM_PASSWORD, VEM_DATABASE for the demo."
        )

    vs = VolcengineMySQLVectorStore.from_params(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        table_name=table_name,
        embed_dim=EMBED_DIM,
        # Override default SSL-related connect args to match the direct
        # PyMySQL usage that already works for you.
        connection_args={"read_timeout": 30},
        ann_index_algorithm="hnsw",
        ann_index_distance="l2",
        ann_m=16,
        ef_search=20,
        perform_setup=True,
        debug=False,
    )

    return vs


def prepare_documents() -> List[Document]:
    docs = [
        Document(
            text=(
                "veDB for MySQL is a cloud-native, high-performance "
                "database service from Volcengine. It provides automatic "
                "scaling, high availability, and excellent performance "
                "for modern applications."
            ),
            metadata={"source": "product_docs", "category": "database"},
        ),
        Document(
            text=(
                "LlamaIndex is a framework for building LLM applications "
                "over your own data sources and knowledge bases."
            ),
            metadata={"source": "framework_docs", "category": "framework"},
        ),
        Document(
            text=(
                "Vector databases enable efficient similarity search for AI "
                "applications by storing high-dimensional embeddings."
            ),
            metadata={"source": "ai_docs", "category": "ai"},
        ),
    ]

    return docs


def run_demo() -> None:
    print("Initializing Ark embedding model...")
    embed_model = ArkEmbedding(client=ARK_CLIENT, model=EMBEDDING_MODEL)
    Settings.embed_model = embed_model

    print("Configuring VolcengineMySQLVectorStore...")
    vector_store = build_vector_store()

    print("Preparing documents...")
    docs = prepare_documents()

    print("Building VectorStoreIndex backed by VolcengineMySQLVectorStore...")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    print("Index built and stored in MySQL vector table.")

    # Demonstrate direct vector-store query
    question = "What is veDB for MySQL?"
    print(f"\n=== Similarity search for: {question!r} ===")

    query_embedding = embed_model.get_text_embedding(question)
    vs_query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=3,
    )

    result = vector_store.query(vs_query)

    print(f"Top {len(result.nodes)} results:")
    for i, node in enumerate(result.nodes, 1):
        sim = result.similarities[i - 1] if result.similarities else None
        if sim is not None:
            print(f"  {i}. similarity={sim:.4f}")
        else:
            print(f"  {i}.")
        print(f"     text={node.get_content()[:120]}...")
        print(f"     metadata={node.metadata}")

    # Simple RAG-style call to Ark LLM using retrieved context
    context_text = "\n\n".join(node.get_content() for node in result.nodes)
    prompt = f"""You are a helpful AI assistant. Use the following context to answer the question.
If you don't know the answer, just say you don't know. Do not make up answers.

Context:
{context_text}

Question: {question}

Answer:"""

    print("\n=== Ark LLM answer ===")
    answer = call_ark_llm(prompt)
    print(answer)

    # Cleanup: drop the demo table after the RAG test finishes
    print("\nCleaning up demo table ...")
    vector_store.drop()
    print("Cleanup complete.")


if __name__ == "__main__":
    run_demo()
