# Usage Pattern

You can create custom LlamaHub Tool Specs and Tools or they can be imported from the `llama-hub` package. They can be plugged into our native agents, or LangChain agents.

## Using with our Agents

To use with our OpenAIAgent,
```python
from llama_index.agent import OpenAIAgent
from llama_hub.tools.gmail.base import GmailToolSpec
from llama_index.tools.function_tool import FunctionTool

# Use a tool spec from Llama-Hub
tool_spec = GmailToolSpec()

# Create a custom tool. Type annotations and docstring are used for the
# tool definition sent to the Function calling API.
def add_numbers(x: int, y: int) -> int:
    """
    Adds the two numbers together and returns the result.
    """
    return x + y

function_tool = FunctionTool.from_defaults(fn=add_numbers)

tools = tool_spec.to_tool_list() + [function_tool]
agent = OpenAIAgent.from_tools(tools, verbose=True)

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