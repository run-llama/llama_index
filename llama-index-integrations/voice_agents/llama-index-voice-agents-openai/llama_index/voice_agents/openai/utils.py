from llama_index.core.tools import BaseTool
from typing import List, Union


def get_tool_by_name(tools: List[BaseTool], name: str) -> Union[BaseTool, None]:
    for tool in tools:
        if tool.metadata.get_name() == name:
            return tool

    return None
