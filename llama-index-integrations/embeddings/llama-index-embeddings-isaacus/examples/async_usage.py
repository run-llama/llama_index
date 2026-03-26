"""Async usage example for Isaacus embeddings."""

import asyncio
import os
from llama_index.embeddings.isaacus import IsaacusEmbedding
from llama_index.core.base.embeddings.base import similarity


async def main():
    """Demonstrate async usage of Isaacus embeddings."""

    # Initialize the embedding model. This assumes the presence of ISAACUS_API_KEY
    # in the host environment
    embedding_model = IsaacusEmbedding()

    # Example legal texts to embed
    texts = [
        "The parties hereby agree to the terms and conditions set forth in this contract.",
        "This agreement shall be governed by the laws of the State of California.",
        "Either party may terminate this contract with 30 days written notice.",
        "The confidentiality provisions shall survive termination of this agreement.",
    ]

    print("Generating embeddings asynchronously for legal texts...")
    print()

    # Get embeddings for individual texts asynchronously
    for i, text in enumerate(texts):
        embedding = await embedding_model.aget_text_embedding(text)
        print(f"Text {i+1}: {text[:60]}...")
        print(f"  Embedding dimension: {len(embedding)}")
        print(f"  First 5 values: {[f'{x:.4f}' for x in embedding[:5]]}")
        print()

    # Get embeddings for all texts at once asynchronously
    print("Getting batch embeddings asynchronously...")
    all_embeddings = await embedding_model.aget_text_embedding_batch(texts)
    print(f"Generated {len(all_embeddings)} embeddings")
    print()

    # Demonstrate query vs document embeddings with async
    print("Demonstrating async query vs document task optimization...")

    # Create a document embedder
    doc_embedder = IsaacusEmbedding(task="retrieval/document")
    doc_embedding = await doc_embedder.aget_text_embedding(texts[0])

    # Get a query embedding (uses retrieval/query task automatically)
    query = "What are the termination terms?"
    query_embedding = await embedding_model.aget_query_embedding(query)

    print(f"Document embedding dimension: {len(doc_embedding)}")
    print(f"Query embedding dimension: {len(query_embedding)}")
    print()

    # Demonstrate similarity (cosine similarity)
    print("Calculating similarities between query and documents...")
    for i, text in enumerate(texts):
        doc_emb = await doc_embedder.aget_text_embedding(text)
        sim = similarity(query_embedding, doc_emb)
        print(f"Similarity to document {i+1}: {sim:.4f}")
        print(f"  Document: {text[:60]}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())
