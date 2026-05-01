# Amazon Bedrock AgentCore Runtime and Tools

This module provides a runtime adapter and tools for deploying and extending LlamaIndex agents with [Amazon Bedrock AgentCore](https://aws.amazon.com/bedrock/agentcore/) -- including managed compute via AgentCore Runtime, sandboxed browser automation, and code execution.

## Prerequisites

- **AWS credentials** configured via environment variables, AWS CLI profile, or IAM role
- **IAM permissions** for `bedrock-agentcore:*` actions (see the [AgentCore documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/agentcore.html) for details)
- **Python 3.9+**

## Installation

(Optional) To run the examples below, first install:

```bash
pip install llama-index llama-index-llms-bedrock-converse
```

Install the main tools package:

```bash
pip install llama-index-tools-aws-bedrock-agentcore
```

## Runtime

The `AgentCoreRuntime` adapter deploys any LlamaIndex agent to [Amazon Bedrock AgentCore Runtime](https://docs.aws.amazon.com/bedrock/latest/userguide/agentcore.html) -- a managed compute platform for AI agents. It wraps `BedrockAgentCoreApp` from the `bedrock-agentcore` SDK, providing the required `POST /invocations` and `GET /ping` endpoints.

### Quick Start

```python
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.tools.aws_bedrock_agentcore import AgentCoreRuntime

llm = BedrockConverse(
    model="us.anthropic.claude-sonnet-4-6-v1",
    region_name="us-west-2",
)
agent = FunctionAgent(llm=llm, tools=[])

# One-liner -- starts uvicorn on port 8080
AgentCoreRuntime.serve(agent)
```

### With Options

```python
runtime = AgentCoreRuntime(
    agent=agent,
    stream=True,  # SSE streaming (default)
    port=8080,  # Required port for AgentCore deployment
    debug=False,
)
runtime.run()
```

### With AgentCore Memory

```python
from llama_index.memory.bedrock_agentcore import (
    AgentCoreMemory,
    AgentCoreMemoryContext,
)

memory = AgentCoreMemory(
    context=AgentCoreMemoryContext(
        memory_id="your-memory-id",
        actor_id="user-123",
    ),
    region_name="us-west-2",
)

# Session ID from the X-Amzn-Bedrock-AgentCore-Runtime-Session-Id header
# is automatically wired to memory
AgentCoreRuntime.serve(agent, memory=memory)
```

### Sending Requests

```bash
# Non-streaming
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, what can you do?"}'

# Streaming (SSE)
curl -N -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, what can you do?"}'
```

The adapter accepts `prompt`, `message`, or `input` as the payload key.

### Streaming Event Types

When `stream=True` (default), the SSE stream emits these event types:

| Event          | Fields                                 | Description               |
| -------------- | -------------------------------------- | ------------------------- |
| `agent_stream` | `delta`, `response`, `thinking_delta`? | Token-by-token LLM output |
| `tool_call`    | `tool_name`, `tool_kwargs`             | Before tool execution     |
| `tool_result`  | `tool_name`, `tool_output`             | After tool execution      |
| `done`         | `response`                             | Final agent response      |
| `error`        | `message`                              | Error during streaming    |

### Testing with ASGI

```python
runtime = AgentCoreRuntime(agent=agent)
app = runtime.app  # BedrockAgentCoreApp (Starlette-based)
# Use with httpx.AsyncClient for testing
```

## Toolspecs

### Browser

The AgentCore `Browser` toolspec provides a set of tools for interacting with web browsers in a secure sandbox environment. It enables your LlamaIndex agents to navigate websites, extract content, click elements, and more.

Included tools:

- `navigate_browser`: Navigate to a URL
- `click_element`: Click on an element using CSS selectors
- `extract_text`: Extract all text from the current webpage
- `extract_hyperlinks`: Extract all hyperlinks from the current webpage
- `get_elements`: Get elements matching a CSS selector
- `navigate_back`: Navigate to the previous page
- `current_webpage`: Get information about the current webpage
- `generate_live_view_url`: Generate a presigned URL for human oversight of a browser session
- `take_control`: Take manual control of a browser session (disables automation)
- `release_control`: Release manual control (re-enables automation)

Lifecycle methods available for programmatic use (not exposed as agent tools):

- `list_browsers`, `create_browser`, `delete_browser`, `get_browser`

You can optionally pass a custom `identifier` for VPC-enabled browser resources:

```python
tool_spec = AgentCoreBrowserToolSpec(
    region="us-west-2",
    identifier="my-custom-browser-id",
)
```

Example usage:

```python
import asyncio
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.tools.aws_bedrock_agentcore import AgentCoreBrowserToolSpec
from llama_index.core.agent.workflow import FunctionAgent

import nest_asyncio

nest_asyncio.apply()  # In case of existing loop (ex. in JupyterLab)


async def main():
    tool_spec = AgentCoreBrowserToolSpec(region="us-west-2")
    tools = tool_spec.to_tool_list()

    llm = BedrockConverse(
        model="us.anthropic.claude-sonnet-4-6-v1",
        region_name="us-west-2",
    )

    agent = FunctionAgent(
        tools=tools,
        llm=llm,
    )

    task = "Go to https://news.ycombinator.com/ and tell me the titles of the top 5 posts."

    response = await agent.run(task)
    print(str(response))

    await tool_spec.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
```

### Code Interpreter

The AgentCore `Code Interpreter` toolspec provides a set of tools for interacting with a secure code interpreter sandbox environment. It enables your LlamaIndex agents to execute code, run shell commands, manage files, and perform computational tasks.

Included tools:

- `execute_code`: Run code in various languages (primarily Python)
- `execute_command`: Run shell commands
- `read_files`: Read content of files in the environment
- `list_files`: List files in directories
- `delete_files`: Remove files from the environment
- `write_files`: Create or update files
- `start_command`: Start long-running commands asynchronously
- `get_task`: Check status of async tasks
- `stop_task`: Stop running tasks
- `upload_file`: Upload a file with an optional semantic description
- `upload_files`: Upload multiple files at once
- `install_packages`: Install Python packages via pip
- `download_file`: Download a file from the sandbox
- `download_files`: Download multiple files from the sandbox
- `clear_context`: Clear all variable state in the Python execution context

Lifecycle methods available for programmatic use (not exposed as agent tools):

- `list_code_interpreters`, `create_code_interpreter`, `delete_code_interpreter`, `get_code_interpreter`

You can optionally pass a custom `identifier` for VPC-enabled code interpreter resources:

```python
tool_spec = AgentCoreCodeInterpreterToolSpec(
    region="us-west-2",
    identifier="my-custom-interpreter-id",
)
```

Example usage:

```python
import asyncio
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.tools.aws_bedrock_agentcore import (
    AgentCoreCodeInterpreterToolSpec,
)
from llama_index.core.agent.workflow import FunctionAgent

import nest_asyncio

nest_asyncio.apply()  # In case of existing loop (ex. in JupyterLab)


async def main():
    tool_spec = AgentCoreCodeInterpreterToolSpec(region="us-west-2")
    tools = tool_spec.to_tool_list()

    llm = BedrockConverse(
        model="us.anthropic.claude-sonnet-4-6-v1",
        region_name="us-west-2",
    )

    agent = FunctionAgent(
        tools=tools,
        llm=llm,
    )

    code_task = "Write a Python function that calculates the factorial of a number and test it."

    code_response = await agent.run(code_task)
    print(str(code_response))

    command_task = "Use terminal CLI commands to: 1) Show the environment's Python version. 2) Show me the list of Python package currently installed in the environment."

    command_response = await agent.run(command_task)
    print(str(command_response))

    await tool_spec.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
```

## Example Notebooks

- [Browser Tool Notebook](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/tools/agentcore_browser.ipynb)
- [Code Interpreter Tool Notebook](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/tools/agentcore_code_interpreter.ipynb)
