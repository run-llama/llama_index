from typing import Any

from llama_index.core.agent.workflow.workflow_events import ToolCallResult
from llama_index.server.api.callbacks.base import EventCallback
from llama_index.server.api.models import SourceNodesEvent


class SourceNodesFromToolCall(EventCallback):
    """
    Extract source nodes from the query tool output.

    Args:
        query_tool_name: The name of the tool that queries the index.
                         default is "query_index"
    """

    def __init__(self, query_tool_name: str = "query_index"):
        self.query_tool_name = query_tool_name

    def transform_tool_call_result(self, event: ToolCallResult) -> SourceNodesEvent:
        source_nodes = event.tool_output.raw_output.source_nodes
        return SourceNodesEvent(nodes=source_nodes)

    async def run(self, event: Any) -> Any:
        if isinstance(event, ToolCallResult):
            if event.tool_name == self.query_tool_name:
                return event, self.transform_tool_call_result(event)
        return event

    def from_default(self, *args, **kwargs) -> "SourceNodesFromToolCall":
        return SourceNodesFromToolCall()
