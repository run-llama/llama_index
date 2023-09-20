"""JSON splitter."""
from typing import List, Optional, Dict, Generator
from llama_index.bridge.pydantic import Field

from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.text_splitter.types import TextSplitter
import json


class JSONSplitter(TextSplitter):
    """Implementation of splitting text for JSON files."""

    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager, exclude=True
    )
    levels_back: int = Field(
        description="The levels deep in the JSON to parse, after the jq_path"
    )
    jq_path: Optional[str] = Field(
        description="A jq path to extract content from the json"
    )

    def __init__(
        self,
        levels_back: int = 0,
        callback_manager: Optional[CallbackManager] = None,
        jq_path: Optional[str] = None,
    ):
        callback_manager = callback_manager or CallbackManager([])

        super().__init__(
            callback_manager=callback_manager,
            levels_back=levels_back,
            jq_path=jq_path,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "JSONSplitter"

    def split_text(self, text: str) -> List[str]:
        if text == "":
            return []

        with self.callback_manager.event(
            CBEventType.CHUNKING, payload={EventPayload.CHUNKS: [text]}
        ):
            return self._split_text(text)

    def _split_text(self, text: str) -> List[str]:
        try:
            import jq
        except ImportError:
            raise ImportError("Please install jq to use the JSONSplitter")

        try:
            data = json.loads(text)
            if self.jq_path:
                data = json.loads(jq.compile(self.jq_path).input(data).text())
        except json.JSONDecodeError:
            # Handle invalid JSON input here
            return []

        documents = []
        if isinstance(data, dict):
            lines = [*self._depth_first_yield(data, self.levels_back, [])]
            documents.append("\n".join(lines))
        elif isinstance(data, list):
            for json_object in data:
                lines = [*self._depth_first_yield(json_object, self.levels_back, [])]
                documents.append("\n".join(lines))
        else:
            raise ValueError("JSON is invalid")

        return documents

    def _depth_first_yield(
        self, json_data: Dict, levels_back: int, path: List[str]
    ) -> Generator[str, None, None]:
        """Do depth first yield of all of the leaf nodes of a JSON.

        Combines keys in the JSON tree using spaces.

        If levels_back is set to 0, prints all levels.

        """
        if isinstance(json_data, dict):
            for key, value in json_data.items():
                new_path = path[:]
                new_path.append(key)
                yield from self._depth_first_yield(value, levels_back, new_path)
        elif isinstance(json_data, list):
            for _, value in enumerate(json_data):
                yield from self._depth_first_yield(value, levels_back, path)
        else:
            new_path = path[-levels_back:]
            new_path.append(str(json_data))
            yield " ".join(new_path)
