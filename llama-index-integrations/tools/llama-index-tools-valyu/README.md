# LlamaIndex Tools Integration: Valyu

This tool connects to [Valyu](https://www.valyu.network/) and its [Exchange Platform](https://exchange.valyu.network/) to easily enable
your agent to search and get content from programmaticly licensed proprietary content and the web.

To begin, you need to obtain an API key on the [Valyu developer dashboard](https://exchange.valyu.network/user/account/api-keys).

## Usage

Here's an example usage of the ValyuToolSpec.

```python
# %pip install llama-index llama-index-core llama-index-tools-valyu

from llama_index.tools.valyu import ValyuToolSpec
from llama_index.agent.openai import OpenAIAgent

valyu_tool = ValyuToolSpec(
    api_key=os.environ["VALYU_API_KEY"],
    max_price=100,  # default is 100
)
agent = OpenAIAgent.from_tools(valyu_tool.to_tool_list())

agent.chat(
    "What are the implications of using different volatility calculation methods (EWMA vs. GARCH) in Value at Risk (VaR) modeling for fixed income portfolios?"
)
```

`context`: Search for a list of documents relating to a natural language query from programmaticly licensed proprietary content and the web.

This loader is designed to be used as a way to load data as a Tool in a Agent.
