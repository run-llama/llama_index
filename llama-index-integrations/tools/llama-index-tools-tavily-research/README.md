# Tavily Research Tool

[Tavily](https://app.tavily.com/) is a robust research API tailored specifically for LLM Agents. It seamlessly integrates with diverse data sources to ensure a superior, relevant research experience.

To begin, you need to obtain an API key on the [Tavily's developer dashboard](https://app.tavily.com/).

## Why Choose Tavily Research API?

1. **Purpose-Built**: Tailored just for LLM Agents, we ensure our features and results resonate with your unique needs. We take care of all the burden in searching, scraping, filtering and extracting information from online sources. All in a single API call!
2. **Versatility**: Beyond just fetching results, Tavily Research API offers precision. With customizable search depths, domain management, and parsing html content controls, you're in the driver's seat.
3. **Performance**: Committed to rapidity and efficiency, our API guarantees real-time outcomes without sidelining accuracy. Please note that we're just getting started, so performance may vary and improve over time.
4. **Integration-friendly**: We appreciate the essence of adaptability. That's why integrating our API with your existing setup is a breeze. You can choose our Python library or a simple API call or any of our supported partners such as [Langchain](https://python.langchain.com/docs/integrations/tools/tavily_search) and [LLamaIndex](https://llamahub.ai/l/tools-tavily).
5. **Transparent & Informative**: Our detailed documentation ensures you're never left in the dark. From setup basics to nuanced features, we've got you covered.

## Usage

This tool has a more extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-tavily-research/examples/tavily.ipynb)

Here's an example usage of the TavilyToolSpec.

```python
from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.core.agent import FunctionCallingAgent
from llama_index.llms.openai import OpenAI

tavily_tool = TavilyToolSpec(
    api_key="your-key",
)
agent = FunctionCallingAgent.from_tools(
    tavily_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4o"),
)

agent.chat("What happened in the latest Burning Man festival?")
```

`search`: Search for relevant dynamic data based on a query. Returns a list of urls and their relevant content.

This loader is designed to be used as a way to load data as a Tool in an Agent.
