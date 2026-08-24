# Finance Agent Tool

This tool connects to various open finance apis and libraries to gather news, earnings information and doing fundamental analysis.

To use this tool, you'll need a few API keys:

- POLYGON_API_KEY -- <https://polygon.io/>
- FINNHUB_API_KEY -- <https://finnhub.io/>
- ALPHA_VANTAGE_API_KEY -- <https://www.alphavantage.co/>
- NEWSAPI_API_KEY -- <https://newsapi.org/>

## Installation

```
pip install llama-index-tools-finance
```

## Usage

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.finance import FinanceAgentToolSpec

POLYGON_API_KEY = ""
FINNHUB_API_KEY = ""
ALPHA_VANTAGE_API_KEY = ""
NEWSAPI_API_KEY = ""
OPENAI_API_KEY = ""

GPT_MODEL_NAME = "gpt-4-0613"


def create_agent(
    polygon_api_key: str,
    finnhub_api_key: str,
    alpha_vantage_api_key: str,
    newsapi_api_key: str,
    openai_api_key: str,
) -> FunctionAgent:
    tool_spec = FinanceAgentToolSpec(
        polygon_api_key,
        finnhub_api_key,
        alpha_vantage_api_key,
        newsapi_api_key,
    )
    llm = OpenAI(temperature=0, model=GPT_MODEL_NAME, api_key=openai_api_key)
    return FunctionAgent(
        tools=tool_spec.to_tool_list(),
        llm=llm,
    )


agent = create_agent(
    POLYGON_API_KEY,
    FINNHUB_API_KEY,
    ALPHA_VANTAGE_API_KEY,
    NEWSAPI_API_KEY,
    OPENAI_API_KEY,
)

response = await agent.run(
    "What happened to AAPL stock on February 19th, 2024?"
)
```

## Adanos Market Sentiment

The package also provides an independent tool spec for the
[Adanos Market Sentiment API](https://adanos.org/). It gives agents structured
stock sentiment from Reddit, X / FinTwit, financial news, and Polymarket, plus
Reddit crypto sentiment.

Create an API key at [adanos.org/register](https://adanos.org/register), then add
the tools to an agent:

```python
import os

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.finance import AdanosMarketSentimentToolSpec

adanos = AdanosMarketSentimentToolSpec(api_key=os.environ["ADANOS_API_KEY"])
agent = FunctionAgent(
    tools=adanos.to_tool_list(),
    llm=OpenAI(model="gpt-4o"),
)

response = await agent.run(
    "Compare AAPL and NVDA Reddit sentiment from 2026-08-01 to 2026-08-07."
)
```

Available functions:

- `get_adanos_stock_sentiment`: sentiment for one stock from a selected source.
- `get_adanos_crypto_sentiment`: Reddit sentiment for one crypto asset.
- `get_adanos_trending_assets`: ranked stock or crypto attention.
- `compare_adanos_asset_sentiment`: side-by-side sentiment for multiple assets.
- `get_adanos_market_sentiment`: an aggregate market snapshot.

Date windows use inclusive UTC `start_date` and `end_date` values. The Free,
Hobby, and Professional plans support up to 30, 90, and 365 days of historical
lookback respectively. See the [API documentation](https://api.adanos.org/docs)
for response fields and current plan limits.

Rate-limit responses raise `requests.HTTPError` with the API message and any
`Retry-After` guidance. The tool does not retry automatically, so callers retain
control over retry timing and quota usage.
