"""Slack tool spec."""

from llama_index.tools.tool_spec.base import BaseToolSpec
from typing import Type
from pydantic import BaseModel

    def get_fn_schema_from_fn_name(self, fn_name: str) -> Type[BaseModel]:
        """Return map from function name."""
        if fn_name == "load_data":
            pass
        elif fn_name == "search_data":
            return NotionSearchDataSchema
        else:
            raise ValueError(f"Invalid function name: {fn_name}")

class SlackToolSpec(BaseToolSpec):
    """Slack tool spec."""
