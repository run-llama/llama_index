# AWS Bedrock AgentCore Tools

This module provides tools for interacting with [AWS Bedrock AgentCore](https://aws.amazon.com/bedrock/agentcore/)'s browser and code interpreter sandbox tools.

## Installation

(Optional) To run the examples below, first install:

```bash
pip install llama-index llama-index-llms-bedrock-converse
```

Install the main tools package:

```bash
pip install llama-index-tools-aws-bedrock-agentcore
```

## Toolspecs

### Browser

The Bedrock AgentCore `Browser` toolspec provides a set of tools for interacting with web browsers in a secure sandbox environment. It enables your LlamaIndex agents to navigate websites, extract content, click elements, and more.

Included tools:

- `navigate_browser`: Navigate to a URL
- `click_element`: Click on an element using CSS selectors
- `extract_text`: Extract all text from the current webpage
- `extract_hyperlinks`: Extract all hyperlinks from the current webpage
- `get_elements`: Get elements matching a CSS selector
- `navigate_back`: Navigate to the previous page
- `current_webpage`: Get information about the current webpage

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
        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
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

The Bedrock AgentCore `code_interpreter` toolspec provides a set of tools interacting with a secure code interpreter sandbox environment. It enables your LlamaIndex agents to execute code, run shell commands, manage files, and perform computational task.

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
        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
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
