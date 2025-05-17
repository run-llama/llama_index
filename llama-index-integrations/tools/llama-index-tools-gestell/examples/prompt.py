"""
Stream a Gestell PROMPT answer and print tokens in real time.
"""

import asyncio
from llama_index.tools.gestell import GestellToolSpec

prompt_text = "Give me a summary of the documents"

async def main():
    gestell = GestellToolSpec()
    response = await gestell.aprompt(prompt_text)
    print(response)

asyncio.run(main())
