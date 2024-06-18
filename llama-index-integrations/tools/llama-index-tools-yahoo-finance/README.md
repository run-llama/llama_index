# Yahoo Finance Tool

This tool connects to Yahoo Finance and allows an Agent to access stock, news, and financial data of a company.

## Usage

Here's an example of how to use this tool:

```python
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec
from llama_index.agent.openai import OpenAIAgent

tool_spec = YahooFinanceToolSpec()
agent = OpenAIAgent.from_tools(tool_spec.to_tool_list())

agent.chat("What is the price of Apple stock?")
agent.chat("What is the latest news about Apple?")
```

The tools available are:

`balance_sheet`: A tool that returns the balance sheet of a company.

`income_statement`: A tool that returns the income statement of a company.

`cash_flow`: A tool that returns the cash flow of a company.

`stock_news`: A tool that returns the latest news about a company.

`stock_basic_info`: A tool that returns basic information about a company including price.

`stock_analyst_recommendations`: A tool that returns analyst recommendations for a company.

This loader is designed to be used as a way to load data as a Tool in a Agent.
