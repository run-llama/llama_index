"""Async usage example for Heroku embeddings."""

import asyncio
import os
from llama_index.embeddings.heroku import HerokuEmbedding


async def main():
    """Demonstrate async usage of Heroku embeddings."""

    # Initialize the embedding model. This assumes the presence of EMBEDDING_MODEL_ID,
    # EMBEDDING_KEY, and EMBEDDING_URL in the host environment
    embedding_model = HerokuEmbedding()

    try:
        # Example texts to embed
        texts = [
            "Hello, world!",
            "This is a test document about artificial intelligence.",
            "Machine learning is a subset of AI.",
            "Natural language processing helps computers understand human language.",
        ]

        print("Generating embeddings asynchronously...")

        # Get embeddings for individual texts asynchronously
        for i, text in enumerate(texts):
            embedding = await embedding_model.aget_text_embedding(text)
            print(f"Text {i+1}: {text[:50]}...")
            print(f"  Embedding dimension: {len(embedding)}")
            print(f"  First 5 values: {embedding[:5]}")
            print()

        # Get embeddings for all texts at once asynchronously
        print("Getting batch embeddings asynchronously...")
        all_embeddings = await embedding_model.aget_text_embedding_batch(texts)
        print(f"Generated {len(all_embeddings)} embeddings")

        # Demonstrate similarity (cosine similarity)
        from llama_index.core.base.embeddings.base import similarity

        print("\nCalculating similarities...")
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = similarity(all_embeddings[i], all_embeddings[j])
                print(f"Similarity between text {i+1} and text {j+1}: {sim:.4f}")

    finally:
        # Clean up async client
        await embedding_model.aclose()


if __name__ == "__main__":
    asyncio.run(main())
