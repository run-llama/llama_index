---
title: AWS Bedrock AgentCore
---

[AWS Bedrock AgentCore](https://aws.amazon.com/bedrock/agentcore/) provides managed infrastructure for building production AI agents. The LlamaIndex integration gives your agents access to sandboxed browser automation, code execution, and persistent memory -- all running in secure AWS environments.

## Installation

```sh
pip install llama-index-tools-aws-bedrock-agentcore
pip install llama-index-memory-bedrock-agentcore
```

**Prerequisites:**

- AWS credentials configured via environment variables, AWS CLI profile, or IAM role
- IAM permissions for `bedrock-agentcore:*` actions (see the [AgentCore documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/agentcore.html))
- Python 3.9+

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
        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
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
        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
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

The `AgentCoreMemory` class provides persistent, managed memory backed by AWS Bedrock AgentCore. It supports short-term chat history via events and long-term memory via semantic search over memory records. Memory is isolated per user via `actor_id`, making it suitable for multi-tenant applications.

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
        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
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
