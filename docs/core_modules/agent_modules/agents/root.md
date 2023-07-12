# Data Agents

## Concept
Data Agents are LLM-powered knowledge workers in LlamaIndex that can intelligently perform various tasks over your data, in both a “read” and “write” function. They are capable of the following:

- Perform automated search and retrieval over different types of data - unstructured, semi-structured, and structured.
- Calling any external service API in a structured fashion, and processing the response + storing it for later.

In that sense, agents are a step beyond our [query engines](/core_modules/query_modules/query_engine/root.md) in that they can not only "read" from a static source of data, but can dynamically ingest and modify data from a variety of different tools.

Building a data agent requires the following core components:

- A reasoning loop
- Tool abstractions

A data agent is initialized with set of APIs, or Tools, to interact with; these APIs can be called by the agent to return information or modify state. Given an input task, the data agent uses a reasoning loop to decide which tools to use, in which sequence, and the parameters to call each tool.

### Reasoning Loop
The reasoning loop depends on the type of agent. We have support for the following agents: 
- OpenAI Function agent (built on top of the OpenAI Function API)
- a ReAct agent (which works across any chat/text completion endpoint).

### Tool Abstractions

You can learn more about our Tool abstractions in our [Tools section](/core_modules/agent_modules/tools/root.md).

### Blog Post

For full details, please check out our detailed [blog post]().


## Usage Pattern

Data agents can be used in the following manner (the example uses the OpenAI Function API)
```python
from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI

# import and define tools
...

# initialize llm
llm = OpenAI(model="gpt-3.5-turbo-0613")

# initialize openai agent
agent = OpenAIAgent.from_tools(tools, llm=llm, verbose=True)
```

See our usage pattern guide for more details.
```{toctree}
---
maxdepth: 1
---
usage_pattern.md
```

## Modules

Learn more about our different agent types in our module guides below.

Also take a look at our [tools section](/core_modules/agent_modules/tools/root.md)!

```{toctree}
---
maxdepth: 2
---
modules.md
```

