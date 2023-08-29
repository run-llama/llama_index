from typing import List

from llama_index.embeddings.base import BaseEmbedding


class MockEmbedding(BaseEmbedding):
    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "MockEmbedding"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        del query
        return [0, 0, 1, 0, 0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        # assume dimensions are 5
        if text == "Hello world.":
            return [1, 0, 0, 0, 0]
        elif text == "This is a test.":
            return [0, 1, 0, 0, 0]
        elif text == "This is another test.":
            return [0, 0, 1, 0, 0]
        elif text == "This is a test v2.":
            return [0, 0, 0, 1, 0]
        elif text == "This is a test v3.":
            return [0, 0, 0, 0, 1]
        elif text == "This is bar test.":
            return [0, 0, 1, 0, 0]
        elif text == "Hello world backup.":
            # this is used when "Hello world." is deleted.
            return [1, 0, 0, 0, 0]
        else:
            return [0, 0, 0, 0, 0]

    def _get_query_embedding(self, query: str) -> List[float]:
        del query  # Unused
        return [0, 0, 1, 0, 0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Mock get text embedding."""
        # assume dimensions are 5
        if text == "Hello world.":
            return [1, 0, 0, 0, 0]
        elif text == "This is a test.":
            return [0, 1, 0, 0, 0]
        elif text == "This is another test.":
            return [0, 0, 1, 0, 0]
        elif text == "This is a test v2.":
            return [0, 0, 0, 1, 0]
        elif text == "This is a test v3.":
            return [0, 0, 0, 0, 1]
        elif text == "This is bar test.":
            return [0, 0, 1, 0, 0]
        elif text == "Hello world backup.":
            # this is used when "Hello world." is deleted.
            return [1, 0, 0, 0, 0]
        else:
            return [0, 0, 0, 0, 0]
