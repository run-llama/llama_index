"""OpenAPI Specification Reader."""

import json
import re
from typing import Any, Dict, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class OpenAPIReader(BaseReader):
    """OpenAPI reader.

    Reads OpenAPI specifications giving options to on how to parse them.

    Args:
        depth (Optional[int]): Depth to dive before splitting the JSON.
        exclude (Optional[List[str]]): JSON paths to exclude, separated by commas by '.'. For example: 'components.pets' will exclude the component 'pets' from the OpenAPI specification. Useful for removing unwanted information from the OpenAPI specification.

    Returns:
        List[Document]: List of documents.

    """

    def __init__(
        self, depth: Optional[int] = 1, exclude: Optional[List[str]] = None
    ) -> None:
        super().__init__()
        self.exclude = exclude
        self.depth = depth

    @classmethod
    def class_name(cls) -> str:
        """Get the name identifier of the class."""
        return "OpenAPIReader"

    def _should_exclude(self, path: str) -> bool:
        """Check if the path should be excluded."""
        return self.exclude and any(
            re.match(exclude_path, path) for exclude_path in self.exclude
        )

    def _build_docs_from_attributes(
        self,
        key: str,
        value: Any,
        extra_info: Dict,
        path: str = "$",
        level: int = 0,
    ) -> List[Document]:
        """Build Documents from the attributes of the OAS JSON."""
        if not path and self._should_exclude(path):
            return []

        if self.depth == level or not isinstance(value, dict):
            return [
                Document(
                    text=f"{key}: {value}", metadata={"json_path": path, **extra_info}
                )
            ]

        return [
            doc
            for k, v in value.items()
            for doc in self._build_docs_from_attributes(
                k, v, extra_info, f"{path}.{key}", level + 1
            )
        ]

    def load_data(
        self, input_file: str, extra_info: Optional[Dict] = {}
    ) -> List[Document]:
        """Load data from the input file."""
        try:
            with open(input_file, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"The file {input_file} is not a valid JSON file.")

        return [
            doc
            for key, value in data.items()
            for doc in self._build_docs_from_attributes(key, value, extra_info)
        ]
