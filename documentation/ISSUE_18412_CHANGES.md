# Issue #18412: Structured Output for AgentWorkflow

## Summary

This document tracks all changes made to implement native structured output support for AgentWorkflow, eliminating the need for extra LLM calls.

## Issue Details

- **Issue Number**: #18412
- **Title**: [Feature Request]: Structured Output for AgentWorkflow
- **Priority**: P0
- **Author**: @pazevedo-hyland
- **Date**: April 9, 2025

## Problem Statement

Users want AgentWorkflow to return structured output (Pydantic models) during tool-calling workflows without requiring an extra LLM call. The existing implementation makes an additional LLM call after agent completion to generate structured output, adding latency and cost.

## Solution Approach

Implemented a "Structured Output as Tool" pattern that:

1. Auto-injects a `StructuredOutputTool` when `output_cls` is provided
2. The tool has `return_direct=True` to capture output immediately
3. Parses tool arguments directly as the structured response
4. Maintains backward compatibility with existing `output_cls` and `structured_output_fn` parameters

## Files Modified

### 1. `llama_index/core/agent/workflow/structured_output.py` (NEW FILE)

**Purpose**: New module containing the StructuredOutputTool implementation

**Classes and Functions**:

- `StructuredOutputTool`: A tool that wraps a Pydantic model for structured output capture
  - `from_output_cls()`: Class method to create the tool from a Pydantic model
  - `call()` / `acall()`: Validate and return structured output
- `STRUCTURED_OUTPUT_TOOL_NAME`: Constant for the tool name ("submit_final_response")
- `is_structured_output_tool()`: Helper to check if a tool is a StructuredOutputTool
- `extract_structured_output_from_tool_result()`: Extract structured data from tool results

### 2. `llama_index/core/agent/workflow/base_agent.py`

**Purpose**: Base agent modifications to support structured output tool injection

**Changes**:

- Added import for `StructuredOutputTool`, `STRUCTURED_OUTPUT_TOOL_NAME`, `extract_structured_output_from_tool_result`
- Added `use_native_structured_output: bool = True` field
- Added `_get_structured_output_tool()` method to create the tool from `output_cls`
- Modified `get_tools()` to accept `include_structured_output_tool` parameter and inject the tool
- Modified `parse_agent_output()` to only use legacy approach when `use_native_structured_output=False`
- Modified `aggregate_tool_results()` to extract structured output when the structured output tool is called

### 3. `llama_index/core/agent/workflow/multi_agent_workflow.py`

**Purpose**: Main workflow orchestration updates

**Changes**:

- Added import for `StructuredOutputTool`, `STRUCTURED_OUTPUT_TOOL_NAME`, `extract_structured_output_from_tool_result`
- Added `use_native_structured_output: bool = True` parameter to `__init__`
- Added `_get_structured_output_tool()` method
- Modified `get_tools()` to inject structured output tool when configured
- Modified `parse_agent_output()` to only use legacy approach when `use_native_structured_output=False`
- Modified `aggregate_tool_results()` to extract structured output from tool results
- Updated `from_tools_or_functions()` to accept `use_native_structured_output` parameter

### 4. `llama_index/core/agent/workflow/__init__.py`

**Purpose**: Export new classes

**Changes**:

- Added export for `StructuredOutputTool`
- Added export for `STRUCTURED_OUTPUT_TOOL_NAME`

### 5. `.gitignore`

**Purpose**: Exclude local documentation folder

**Changes**:

- Added `/documentation/` to gitignore

### 6. `tests/agent/workflow/test_structured_output_tool.py` (NEW FILE)

**Purpose**: Comprehensive tests for the new structured output tool functionality

**Test Classes**:

- `TestStructuredOutputTool`: Unit tests for StructuredOutputTool class
- `TestExtractStructuredOutput`: Tests for the extraction helper function
- `TestAgentWithNativeStructuredOutput`: Integration tests for agents with native structured output
- `TestBackwardCompatibility`: Tests ensuring backward compatibility with legacy mode

## API Changes

### New Parameters

- `use_native_structured_output: bool = True` - When True, uses the tool-based approach. When False, falls back to extra LLM call.

### New Exports

- `StructuredOutputTool`: The tool class for structured output
- `STRUCTURED_OUTPUT_TOOL_NAME`: The constant tool name

### Behavior Changes

- When `output_cls` is set and `use_native_structured_output=True` (default):
  - A `StructuredOutputTool` is automatically added to the agent's tools
  - The agent can call this tool on its final turn to produce structured output
  - No extra LLM call is needed, reducing latency and cost
- When `use_native_structured_output=False`:
  - Falls back to the legacy behavior with an extra LLM call

## Backward Compatibility

- Existing code using `output_cls` continues to work (now with native mode by default)
- Existing code using `structured_output_fn` continues to work unchanged
- Set `use_native_structured_output=False` to explicitly use the old behavior

## Testing

Run tests with:

```bash
cd llama-index-core
uv run -- pytest tests/agent/workflow/test_structured_output_tool.py -v
```

## Example Usage

```python
from pydantic import BaseModel, Field
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent


class MathResult(BaseModel):
    operation: str = Field(description="The operation performed")
    result: int = Field(description="The result of the operation")


def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y


# Native structured output is enabled by default
workflow = AgentWorkflow.from_tools_or_functions(
    tools_or_functions=[multiply],
    output_cls=MathResult,
)

result = await workflow.run("What is 30 * 60?")
print(
    result.structured_response
)  # {'operation': 'multiplication', 'result': 1800}

# To use legacy mode with extra LLM call:
workflow_legacy = AgentWorkflow.from_tools_or_functions(
    tools_or_functions=[multiply],
    output_cls=MathResult,
    use_native_structured_output=False,  # Use legacy approach
)
```

## Implementation Notes

1. The `StructuredOutputTool` is designed with `return_direct=True` so it immediately returns when called, stopping the agent loop.

2. The tool's description includes the JSON schema of the output class, helping the LLM understand the expected format.

3. When the structured output tool is called, `aggregate_tool_results()` detects it and extracts the structured response from the tool arguments.

4. The implementation preserves all existing functionality while adding the new native mode as the default.

## Performance Implications

- **Native mode (default)**: Eliminates one LLM call, reducing latency and API costs
- **Legacy mode**: Same behavior as before (extra LLM call for structured output)
