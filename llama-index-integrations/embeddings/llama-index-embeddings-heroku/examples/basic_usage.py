"""Basic usage example for Heroku embeddings."""

import os
from llama_index.embeddings.heroku import HerokuEmbedding


def main():
    """Demonstrate basic usage of Heroku embeddings."""

    # Initialize the embedding model
    embedding_model = HerokuEmbedding()

    # Example texts to embed
    texts = [
        "Hello, world!",
        "This is a test document about artificial intelligence.",
        "Machine learning is a subset of AI.",
        "Natural language processing helps computers understand human language.",
    ]

    print("Generating embeddings...")

    # Get embeddings for individual texts
    for i, text in enumerate(texts):
        embedding = embedding_model.get_text_embedding(text)
        print(f"Text {i+1}: {text[:50]}...")
        print(f"  Embedding dimension: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")
        print()

    # Get embeddings for all texts at once
    print("Getting batch embeddings...")
    all_embeddings = embedding_model.get_text_embeddings(texts)
    print(f"Generated {len(all_embeddings)} embeddings")

    # Demonstrate similarity (cosine similarity)
    from llama_index.core.embeddings import similarity

    print("\nCalculating similarities...")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = similarity(all_embeddings[i], all_embeddings[j])
            print(f"Similarity between text {i+1} and text {j+1}: {sim:.4f}")


if __name__ == "__main__":
    main()
