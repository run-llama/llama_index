"""
Run a Gestell SEARCH query and show the top results.
"""

import asyncio
import os
from llama_index.tools.gestell import GestellToolSpec

query = "Give me a summary of all documents"

async def main():

    gestell = GestellToolSpec()
    results = await gestell.asearch(query)

    print(f"\nTop {len(results)} hits:")
    for i, result in enumerate(results):
        print(f"\n[{i+1}] Citation: {result.citation}")
        print(f"Reason: {result.reason}")
        print(f"Content preview: {result.content[:200]}...")

if __name__ == "__main__":
    asyncio.run(main())
