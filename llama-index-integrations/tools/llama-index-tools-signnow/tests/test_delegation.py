from unittest.mock import MagicMock
from typing import List
import pytest
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.tools.signnow.base import SignNowMCPToolSpec


def test_to_tool_list_delegates() -> None:
    spec = SignNowMCPToolSpec.__new__(SignNowMCPToolSpec)
    spec._mcp_spec = MagicMock()
    # Stub to_tool_list to return correct type
    dummy_tool = FunctionTool.from_defaults(fn=lambda: None, name="ok")
    spec._mcp_spec.to_tool_list.return_value = [dummy_tool]
    result = spec.to_tool_list()
    assert isinstance(result, list)
    assert all(isinstance(t, FunctionTool) for t in result)


@pytest.mark.asyncio
async def test_to_tool_list_async_delegates() -> None:
    spec = SignNowMCPToolSpec.__new__(SignNowMCPToolSpec)
    spec._mcp_spec = MagicMock()
    dummy_tool = FunctionTool.from_defaults(fn=lambda: None, name="ok")

    async def _async_return() -> List[FunctionTool]:
        return [dummy_tool]

    spec._mcp_spec.to_tool_list_async = _async_return
    result = await spec.to_tool_list_async()
    assert isinstance(result, list)
    assert all(isinstance(t, FunctionTool) for t in result)
