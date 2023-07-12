# Usage Pattern

Our Tool Specs and Tools can be imported from the `llama-hub` package. They can be plugged into our native agents, or LangChain agents.

To use with our OpenAIAgent,
```python
from llama_index.agent import OpenAIAgent
from llama_hub.tools.tool_spec.gmail.base import GmailToolSpec

tool_spec = GmailToolSpec()
agent = OpenAIAgent.from_tools(tool_spec.to_tool_list(), verbose=True)

# use agent
agent.chat("Can you create a new email to helpdesk and support @example.com about a service outage")
```

Full Tool details can be found on our [LlamaHub](llamahub.ai) page. Each tool contains a "Usage" section showing how that tool can be used.


## Using with LangChain
To use with a LangChain agent, simply convert tools to LangChain tools with `to_langchain_tool()`.

```python
tools = tool_spec.to_tool_list()
langchain_tools = [t.to_langchain_tool() for t in tools]
# plug into LangChain agent
from langchain.agents import initialize_agent

agent_executor = initialize_agent(
    langchain_tools, llm, agent="conversational-react-description", memory=memory
)

```