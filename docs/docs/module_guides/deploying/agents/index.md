# Agents

## Concept

Data Agents are LLM-powered knowledge workers in LlamaIndex that can intelligently perform various tasks over your data, in both a “read” and “write” function. They are capable of the following:

- Perform automated search and retrieval over different types of data - unstructured, semi-structured, and structured.
- Calling any external service API in a structured fashion, and processing the response + storing it for later.

In that sense, agents are a step beyond our [query engines](../query_engine/index.md) in that they can not only "read" from a static source of data, but can dynamically ingest and modify data from a variety of different tools.

Building a data agent requires the following core components:

- A reasoning loop
- Tool abstractions

A data agent is initialized with set of APIs, or Tools, to interact with; these APIs can be called by the agent to return information or modify state. Given an input task, the data agent uses a reasoning loop to decide which tools to use, in which sequence, and the parameters to call each tool.

### Reasoning Loop

The reasoning loop depends on the type of agent. We have support for the following agents:

- Function Calling Agents (integrates with any function calling LLM)
- ReAct agent (works across any chat/text completion endpoint).
- "Advanced Agents": [LLMCompiler](https://llamahub.ai/l/llama-packs/llama-index-packs-agents-llm-compiler?from=), [Chain-of-Abstraction](https://llamahub.ai/l/llama-packs/llama-index-packs-agents-coa?from=), [Language Agent Tree Search](https://llamahub.ai/l/llama-packs/llama-index-packs-agents-lats?from=), and more.

### Tool Abstractions

You can learn more about our Tool abstractions in our [Tools section](./tools.md).

### Blog Post

For full details, please check out our detailed [blog post](https://medium.com/llamaindex-blog/data-agents-eed797d7972f).

### Lower-level API: Step-Wise Execution

By default, our agents expose `query` and `chat` functions that will execute a user-query end-to-end.

We also offer a **lower-level API** allowing you to perform step-wise execution of an agent. This gives you much more control in being able to create tasks, and analyze + act upon the input/output of each step within a task.

Check out [our guide](agent_runner.md).

## Usage Pattern

Data agents can be used in the following manner (the example uses the OpenAI Function API)

```python
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI

# import and define tools
...

# initialize llm
llm = OpenAI(model="gpt-3.5-turbo-0613")

# initialize openai agent
agent = OpenAIAgent.from_tools(tools, llm=llm, verbose=True)
```

See our [usage pattern guide](usage_pattern.md) for more details.

## Modules

Learn more about our different agent types and use cases in our [module guides](./modules.md).

We also have a [lower-level api guide](./agent_runner.md) for agent runenrs and workers.

Also take a look at our [tools section](./tools.md)!
