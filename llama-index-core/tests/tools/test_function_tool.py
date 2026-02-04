import pytest
import asyncio
from typing import AsyncGenerator, Iterator, Any
from llama_index.core.tools import FunctionTool
from llama_index.core.tools.types import ToolOutput

# --- Mock Functions ---

def sync_fn(input: str) -> str:
    return f"processed {input}"

def sync_generator_fn(input: str) -> Iterator[str]:
    yield "start"
    yield "processing"
    yield f"done {input}"

async def async_fn(input: str) -> str:
    return f"processed {input}"

async def async_generator_fn(input: str) -> AsyncGenerator[dict, None]:
    yield {"status": "loading"}
    yield {"status": "processing"}
    yield {"result": f"done {input}"}

# --- Tests ---

def test_sync_fn_call():
    tool = FunctionTool.from_defaults(fn=sync_fn)
    output = tool.call(input="test")
    assert output.content == "processed test"

def test_sync_generator_fn_call():
    # Should consume generator and return last item
    tool = FunctionTool.from_defaults(fn=sync_generator_fn)
    output = tool.call(input="test")
    assert output.content == "done test"

@pytest.mark.asyncio
async def test_async_fn_acall():
    tool = FunctionTool.from_defaults(async_fn=async_fn)
    output = await tool.acall(input="test")
    assert output.content == "processed test"

@pytest.mark.asyncio
async def test_async_generator_fn_acall():
    # Should consume async generator and return last item
    tool = FunctionTool.from_defaults(async_fn=async_generator_fn)
    output = await tool.acall(input="test")
    assert output.raw_output == {"result": "done test"}

@pytest.mark.asyncio
async def test_sync_generator_fn_acall_parity():
    # acall should also handle sync generators (wrapped in sync_to_async or just returning iterator)
    # If we pass a pure sync generator function to async_fn (unlikely) or if wrapped fn returns iterator.
    
    # Case 1: FunctionTool wrapping a sync generator as 'fn', called via 'acall'
    # FunctionTool wraps 'fn' into 'async_fn' using 'async_to_sync'? No, 'fn' -> 'async_fn' uses 'sync_to_async'.
    tool = FunctionTool.from_defaults(fn=sync_generator_fn)
    output = await tool.acall(input="test")
    assert output.content == "done test"

@pytest.mark.asyncio
async def test_async_generator_fn_acall_stream():
    tool = FunctionTool.from_defaults(async_fn=async_generator_fn)
    
    outputs = []
    async for output in tool.acall_stream(input="stream"):
        outputs.append(output)
    
    assert len(outputs) == 3
    assert outputs[0].is_preliminary is True
    assert outputs[0].raw_output == {"status": "loading"}
    assert outputs[1].is_preliminary is True
    assert outputs[1].raw_output == {"status": "processing"}
    assert outputs[2].is_preliminary is False
    assert outputs[2].raw_output == {"result": "done stream"}

@pytest.mark.asyncio
async def test_async_fn_acall_stream():
    # Should work and yield single result
    tool = FunctionTool.from_defaults(async_fn=async_fn)
    
    outputs = []
    async for output in tool.acall_stream(input="stream"):
        outputs.append(output)
    
    assert len(outputs) == 1
    assert outputs[0].is_preliminary is False
    assert outputs[0].content == "processed stream"
