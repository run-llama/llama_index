# LlamaIndex Memory Integration: Bedrock AgentCore

## Installation

To install the required package, run:

```bash
%pip install llama-index-memory-bedrock-agentcore
```

## Bedrock AgentCore Setup Pre-Requisites

1. AWS account with Bedrock AgentCore access
2. Configured AWS credentials (boto3)
3. Created memory resource in AWS Bedrock AgentCore
4. Required IAM permissions:
   1. bedrock-agentcore:CreateEvent
   2. bedrock-agentcore:ListEvents
   3. bedrock-agentcore:RetrieveMemories

## Sample Usage

1. Create an instance of AgentCoreMemoryContext to setup the memory resources that you will need for building an agent.
   1. Actor id → This is a required field and it is the identifier of the actor (could be an agent or the end-user).
   2. Memory id → This is a required field and it is the identifier of the memory store.
   3. Session id → This is a required field and it is the unique identifier of a particular conversation.
   4. Namespace → This is an optional field and it is used to determine how to extract long term memories. By default it will use “/” as the namespace.
   5. Memory strategy id → This is an optional field and it is the identifier for a memory strategy.

```python
context = AgentCoreMemoryContext(
    actor_id="<INSERT_HERE>",
    memory_id="<INSERT_HERE>",
    session_id="<INSERT_HERE>",
    namespace="<INSERT_HERE>",
    memory_strategy_id="<INSERT_HERE>",
)
agentcore_memory = AgentCoreMemory(context=context)
```

2. This is an example of how to create a FunctionAgent in LlamaIndex. This sample adds a tool & the Claude Sonnet 4 LLM to the agent’s initialization.

If you would like to use this tool, the run the following command

```bash
%pip install llama-index-tools-yahoo-finance
```

```python
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec

llm = BedrockConverse(model="us.anthropic.claude-sonnet-4-20250514-v1:0")

finance_tool_spec = YahooFinanceToolSpec()
agent = FunctionAgent(
    tools=finance_tool_spec.to_tool_list(),
    llm=llm,
)
```

Here's a simpler example that doesn't utilize a third party tool

```python
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import FunctionAgent


def call_fn(name: str):
    """Call the provided name.
    Args:
        name: str (Name of the person)
    """
    print(f"Calling... {name}")


def email_fn(name: str):
    """Email the provided name.
    Args:
        name: str (Name of the person)
    """
    print(f"Emailing... {name}")


call_tool = FunctionTool.from_defaults(fn=call_fn)
email_tool = FunctionTool.from_defaults(fn=email_fn)

agent = FunctionAgent(
    tools=[call_tool, email_tool],
    llm=llm,
)
```

3. Invoke the agent to start conversations

This sample will invoke the tool and store the events in AgentCore Memory

```python
response = await agent.run(
    "What is the stock price of Amazon?", memory=agentcore_memory
)
```

After events are stored, you can then prompt the agent to answer any queries based on the memory records

```python
response = await agent.run(
    "What stock prices have I asked for?", memory=agentcore_memory
)
```

## References

- [Bedrock AgentCore Memory Documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory-getting-started.html)
