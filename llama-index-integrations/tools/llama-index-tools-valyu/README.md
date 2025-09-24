# LlamaIndex Tools Integration: Valyu

This tool connects to [Valyu](https://www.valyu.network/) and its [Exchange Platform](https://platform.valyu.network/) to easily enable
your agent to search and get content from programmatically licensed proprietary content and the web using Valyu's deep search API.

To begin, you need to obtain an API key on the [Valyu developer dashboard](https://platform.valyu.network/user/account/api-keys). You can also use the SDK without an API key by setting the `VALYU_API_KEY` environment variable.

## Usage

Here's an example usage of the ValyuToolSpec.

```python
# %pip install llama-index llama-index-core llama-index-tools-valyu

from llama_index.tools.valyu import ValyuToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
import os

valyu_tool = ValyuToolSpec(
    api_key=os.environ["VALYU_API_KEY"],
    max_price=100,  # default is 100
)
agent = FunctionAgent(
    tools=valyu_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

print(
    await agent.run(
        "What are the implications of using different volatility calculation methods (EWMA vs. GARCH) in Value at Risk (VaR) modeling for fixed income portfolios?"
    )
)

# You can also search directly with the new parameters
results = valyu_tool.search(
    query="artificial intelligence trends 2024",
    included_sources=[
        "arxiv.org",
        "nature.com",
    ],  # Only search academic sources
    response_length="medium",  # 50k characters per result
    max_num_results=3,
    relevance_threshold=0.5,
)
```

`search`: Search and retrieve relevant content from proprietary and public sources using Valyu's deep search. Supports filtering by:

- Search type ("all", "proprietary", or "web")
- Relevance threshold
- Date ranges (start_date, end_date)
- Source filtering (included_sources, excluded_sources)
- Response length (integer for character count or preset values: "short" 25k, "medium" 50k, "large" 100k, "max" full content)

This loader is designed to be used as a way to load data as a Tool in a Agent.
