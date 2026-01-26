import pytest
import asyncio
from typing import AsyncGenerator
from llama_index.core.tools import FunctionTool
from llama_index.core.tools.types import ToolOutput

async def normal_async_fn(input: str) -> str:
    return f"processed {input}"

async def streaming_async_fn(input: str) -> AsyncGenerator[dict, None]:
    yield {"status": "loading", "progress": 0}
    yield {"status": "processing", "progress": 50}
    yield {"status": "done", "result": f"processed {input}"}

@pytest.mark.asyncio
async def test_normal_async_fn_acall():
    tool = FunctionTool.from_defaults(async_fn=normal_async_fn)
    output = await tool.acall(input="test")
    assert output.content == "processed test"

@pytest.mark.asyncio
async def test_generator_async_fn_acall_backward_compat():
    # Should consume generator and return last item
    tool = FunctionTool.from_defaults(async_fn=streaming_async_fn)
    output = await tool.acall(input="test")
    # The last item yielded is the dict
    assert output.raw_output == {"status": "done", "result": "processed test"}

@pytest.mark.asyncio
async def test_generator_async_fn_acall_stream():
    tool = FunctionTool.from_defaults(async_fn=streaming_async_fn)
    
    outputs = []
    async for output in tool.acall_stream(input="stream"):
        outputs.append(output)
    
    assert len(outputs) == 3
    
    # Check first output (preliminary)
    assert outputs[0].is_preliminary is True
    assert outputs[0].raw_output == {"status": "loading", "progress": 0}
    
    # Check second output (preliminary)
    assert outputs[1].is_preliminary is True
    assert outputs[1].raw_output == {"status": "processing", "progress": 50}
    
    # Check last output (final)
    assert outputs[2].is_preliminary is False
    assert outputs[2].raw_output == {"status": "done", "result": "processed stream"}

@pytest.mark.asyncio
async def test_normal_async_fn_acall_stream():
    # Should work and yield single result
    tool = FunctionTool.from_defaults(async_fn=normal_async_fn)
    
    outputs = []
    async for output in tool.acall_stream(input="stream"):
        outputs.append(output)
        
    assert len(outputs) == 1
    assert outputs[0].is_preliminary is False
    assert outputs[0].content == "processed stream"
