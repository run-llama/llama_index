"""Basic usage example for Isaacus embeddings."""

import os
from llama_index.embeddings.isaacus import IsaacusEmbedding
from llama_index.core.base.embeddings.base import similarity


def main():
    """Demonstrate basic usage of Isaacus embeddings."""

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

    print("Generating embeddings for legal texts...")
    print()

    # Get embeddings for individual texts
    for i, text in enumerate(texts):
        embedding = embedding_model.get_text_embedding(text)
        print(f"Text {i+1}: {text[:60]}...")
        print(f"  Embedding dimension: {len(embedding)}")
        print(f"  First 5 values: {[f'{x:.4f}' for x in embedding[:5]]}")
        print()

    # Get embeddings for all texts at once (batch processing)
    print("Getting batch embeddings...")
    all_embeddings = embedding_model.get_text_embedding_batch(texts)
    print(f"Generated {len(all_embeddings)} embeddings")
    print()

    # Demonstrate query vs document embeddings
    print("Demonstrating query vs document task optimization...")

    # Create a document embedder
    doc_embedder = IsaacusEmbedding(task="retrieval/document")
    doc_embedding = doc_embedder.get_text_embedding(texts[0])

    # Get a query embedding (uses retrieval/query task automatically)
    query = "What are the termination terms?"
    query_embedding = embedding_model.get_query_embedding(query)

    print(f"Document embedding dimension: {len(doc_embedding)}")
    print(f"Query embedding dimension: {len(query_embedding)}")
    print()

    # Demonstrate similarity (cosine similarity)
    print("Calculating similarities between query and documents...")
    for i, text in enumerate(texts):
        doc_emb = doc_embedder.get_text_embedding(text)
        sim = similarity(query_embedding, doc_emb)
        print(f"Similarity to document {i+1}: {sim:.4f}")
        print(f"  Document: {text[:60]}...")
        print()


if __name__ == "__main__":
    main()
