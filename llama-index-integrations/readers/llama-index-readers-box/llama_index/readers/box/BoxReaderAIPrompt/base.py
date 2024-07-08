import logging
from typing import List, Any

from llama_index.core.readers.base import (
    BaseReader,
)
from llama_index.core.schema import Document

from box_sdk_gen import (
    BoxClient,
)

logger = logging.getLogger(__name__)


class BoxReaderAIPrompt(BaseReader):
    _box_client: BoxClient

    @classmethod
    def class_name(cls) -> str:
        return "BoxReaderAIPrompt"

    def __init__(self, box_client: BoxClient):
        self._box_client = box_client

    def load_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        return super().load_data(*args, **load_kwargs)
