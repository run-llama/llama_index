"""
Stream a Gestell PROMPT answer and print tokens in real time.
"""

import asyncio
from llama_index.packs.gestell_sdk import GestellSDKPack

prompt_text = "Give me a summary of the documents"

async def main():
    pack = GestellSDKPack()
    async for chunk in pack.aprompt(prompt_text):
        print(chunk, end="", flush=True)

asyncio.run(main())
