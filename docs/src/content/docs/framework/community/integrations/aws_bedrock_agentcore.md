---
title: Amazon Bedrock AgentCore Runtime and Tools
---

[Amazon Bedrock AgentCore](https://aws.amazon.com/bedrock/agentcore/) provides managed infrastructure for deploying and running production AI agents. The LlamaIndex integration lets you deploy agents to AgentCore Runtime with a single line of code, and gives your agents access to sandboxed browser automation, code execution, and persistent memory -- all running in secure AWS environments.

## Installation

```sh
pip install llama-index-tools-aws-bedrock-agentcore
pip install llama-index-memory-bedrock-agentcore
```

**Prerequisites:**

- AWS credentials configured via environment variables, AWS CLI profile, or IAM role
- IAM permissions for `bedrock-agentcore:*` actions (see the [AgentCore documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/agentcore.html))
- Python 3.9+

## Runtime

The `AgentCoreRuntime` adapter deploys any LlamaIndex agent to [Amazon Bedrock AgentCore Runtime](https://docs.aws.amazon.com/bedrock/latest/userguide/agentcore.html) -- a managed compute platform for AI agents. It wraps `BedrockAgentCoreApp` from the `bedrock-agentcore` SDK, providing the required `POST /invocations` and `GET /ping` endpoints with automatic SSE streaming support.

```python
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.tools.aws_bedrock_agentcore import (
    AgentCoreBrowserToolSpec,
    AgentCoreRuntime,
)

tool_spec = AgentCoreBrowserToolSpec(region="us-west-2")
tools = tool_spec.to_tool_list()

llm = BedrockConverse(
    model="us.anthropic.claude-sonnet-4-6-v1",
    region_name="us-west-2",
)

agent = FunctionAgent(tools=tools, llm=llm)

# One-liner -- starts uvicorn on port 8080
AgentCoreRuntime.serve(agent)
```

Or with more control:

```python
runtime = AgentCoreRuntime(
    agent=agent,
    stream=True,  # SSE streaming (default)
    port=8080,  # Required port for AgentCore deployment
    debug=False,  # Enable debug logging
    memory=memory,  # Optional AgentCoreMemory instance
)
runtime.run()
```

Session IDs from the `X-Amzn-Bedrock-AgentCore-Runtime-Session-Id` header are automatically propagated to `AgentCoreMemory` when provided.

When streaming, the SSE stream emits these event types:

| Event          | Fields                                 | Description               |
| -------------- | -------------------------------------- | ------------------------- |
| `agent_stream` | `delta`, `response`, `thinking_delta`? | Token-by-token LLM output |
| `tool_call`    | `tool_name`, `tool_kwargs`             | Before tool execution     |
| `tool_result`  | `tool_name`, `tool_output`             | After tool execution      |
| `done`         | `response`                             | Final agent response      |
| `error`        | `message`                              | Error during streaming    |

## Browser Tools

The `AgentCoreBrowserToolSpec` gives agents the ability to navigate websites, extract content, click elements, and interact with web pages in a secure sandboxed browser.

**Available tools:** `navigate_browser`, `click_element`, `extract_text`, `extract_hyperlinks`, `get_elements`, `navigate_back`, `current_webpage`, `generate_live_view_url`, `take_control`, `release_control`

**Lifecycle methods** (programmatic use): `list_browsers`, `create_browser`, `delete_browser`, `get_browser`

```python
import asyncio
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.tools.aws_bedrock_agentcore import AgentCoreBrowserToolSpec
from llama_index.core.agent.workflow import FunctionAgent


async def main():
    tool_spec = AgentCoreBrowserToolSpec(region="us-west-2")
    tools = tool_spec.to_tool_list()

    llm = BedrockConverse(
        model="us.anthropic.claude-sonnet-4-6-v1",
        region_name="us-west-2",
    )

    agent = FunctionAgent(tools=tools, llm=llm)

    response = await agent.run(
        "Go to https://news.ycombinator.com/ and tell me the titles of the top 5 posts."
    )
    print(str(response))

    await tool_spec.cleanup()


asyncio.run(main())
```

You can optionally pass a custom `identifier` for VPC-enabled browser resources:

```python
tool_spec = AgentCoreBrowserToolSpec(
    region="us-west-2",
    identifier="my-custom-browser-id",
)
```

## Code Interpreter Tools

The `AgentCoreCodeInterpreterToolSpec` gives agents the ability to execute Python code, run shell commands, and manage files in a secure sandbox with up to 8-hour sessions.

**Available tools:** `execute_code`, `execute_command`, `read_files`, `list_files`, `delete_files`, `write_files`, `start_command`, `get_task`, `stop_task`, `upload_file`, `upload_files`, `install_packages`, `download_file`, `download_files`, `clear_context`

**Lifecycle methods** (programmatic use): `list_code_interpreters`, `create_code_interpreter`, `delete_code_interpreter`, `get_code_interpreter`

```python
import asyncio
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.tools.aws_bedrock_agentcore import (
    AgentCoreCodeInterpreterToolSpec,
)
from llama_index.core.agent.workflow import FunctionAgent


async def main():
    tool_spec = AgentCoreCodeInterpreterToolSpec(region="us-west-2")
    tools = tool_spec.to_tool_list()

    llm = BedrockConverse(
        model="us.anthropic.claude-sonnet-4-6-v1",
        region_name="us-west-2",
    )

    agent = FunctionAgent(tools=tools, llm=llm)

    response = await agent.run(
        "Write a Python function that calculates the factorial of a number and test it."
    )
    print(str(response))

    await tool_spec.cleanup()


asyncio.run(main())
```

You can optionally pass a custom `identifier` for VPC-enabled code interpreter resources:

```python
tool_spec = AgentCoreCodeInterpreterToolSpec(
    region="us-west-2",
    identifier="my-custom-interpreter-id",
)
```

## Memory

The `AgentCoreMemory` class provides persistent, managed memory backed by Amazon Bedrock AgentCore. It supports short-term chat history via events and long-term memory via semantic search over memory records. Memory is isolated per user via `actor_id`, making it suitable for multi-tenant applications.

> **Note:** You must first create a memory resource in the AgentCore console or via the AWS SDK to obtain a `memory_id`. See the [AgentCore documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/agentcore.html) for setup instructions.

```python
import asyncio
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.memory.bedrock_agentcore import (
    AgentCoreMemory,
    AgentCoreMemoryContext,
)


async def main():
    memory = AgentCoreMemory(
        context=AgentCoreMemoryContext(
            memory_id="your-memory-id",  # from AgentCore console or API
            actor_id="user-123",
            session_id="session-456",
            namespace="/",
        ),
        region_name="us-west-2",
    )

    llm = BedrockConverse(
        model="us.anthropic.claude-sonnet-4-6-v1",
        region_name="us-west-2",
    )

    agent = FunctionAgent(llm=llm, tools=[])

    # Memory persists across agent runs
    response = await agent.run("My name is Alice.", memory=memory)
    print(str(response))

    response = await agent.run("What is my name?", memory=memory)
    print(str(response))


asyncio.run(main())
```

## Example Notebooks

- [Browser Tool Notebook](/python/examples/tools/bedrock_agentcore_browser)
- [Code Interpreter Tool Notebook](/python/examples/tools/bedrock_agentcore_code_interpreter)

## Resources

- [AWS Bedrock AgentCore Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/agentcore.html)
- [Tools Package on PyPI](https://pypi.org/project/llama-index-tools-aws-bedrock-agentcore/)
- [Memory Package on PyPI](https://pypi.org/project/llama-index-memory-bedrock-agentcore/)
