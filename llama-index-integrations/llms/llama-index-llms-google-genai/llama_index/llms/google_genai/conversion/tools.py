from __future__ import annotations

from typing import TYPE_CHECKING

import google.genai
import google.genai.types as types
from google.genai import _transformers

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool


class ToolSchemaConverter:
    """Convert tool schemas to Gemini function declarations."""

    def __init__(self, *, client: google.genai.Client) -> None:
        self._client = client

    def to_function_declaration(self, tool: "BaseTool") -> types.FunctionDeclaration:
        """Convert a tool schema into a Gemini function declaration."""
        if not tool.metadata.fn_schema:
            raise ValueError("fn_schema is missing")

        root_schema = _transformers.t_schema(self._client, tool.metadata.fn_schema)

        description_parts = tool.metadata.description.split("\n", maxsplit=1)
        if len(description_parts) > 1:
            description = description_parts[-1]
        elif len(description_parts) == 1:
            description = description_parts[0]
        else:
            description = None

        return types.FunctionDeclaration(
            description=description,
            name=tool.metadata.name,
            parameters=root_schema,
        )
