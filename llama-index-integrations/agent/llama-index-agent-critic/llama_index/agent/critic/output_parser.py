import asyncio
from typing import Any
from llama_index.core.output_parsers.base import BaseOutputParser


class CritiqueOutputParser(BaseOutputParser):
    async def aparse(self, output: str) -> Any:
        ...

    def parse(self, output: str) -> Any:
        return asyncio.run(self.aparse(output=output))


class CorrectOutputParser(BaseOutputParser):
    async def aparse(self, output: str) -> Any:
        ...

    def parse(self, output: str) -> Any:
        return asyncio.run(self.aparse(output=output))
