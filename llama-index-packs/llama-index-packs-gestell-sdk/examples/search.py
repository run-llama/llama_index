"""
Run a Gestell SEARCH query and show the top results.
"""

import asyncio
import os
from llama_index.packs.gestell_sdk import GestellSDKPack

query = "Give me a summary of all documents"

async def main():
    # Get credentials from environment variables with fallback to test values
    api_key = os.getenv("GESTELL_API_KEY", "test")
    collection_id = os.getenv("GESTELL_COLLECTION_ID", "test")

    pack = GestellSDKPack(api_key=api_key, collection_id=collection_id)
    results = await pack.asearch(query)

    print(f"\nTop {len(results)} hits:")
    for i, result in enumerate(results):
        print(f"\n[{i+1}] Citation: {result.citation}")
        print(f"Reason: {result.reason}")
        print(f"Content preview: {result.content[:200]}...")

if __name__ == "__main__":
    asyncio.run(main())
