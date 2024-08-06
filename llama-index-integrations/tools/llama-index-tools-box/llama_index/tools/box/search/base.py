from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class BoxSearchToolSpec(BaseToolSpec):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def search(self, query: str) -> Document:
        pass
