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
