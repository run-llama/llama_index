# Stock Market Data Query Engine Pack

Query and retrieve historical market data for a list of stock tickers. It utilizes Yahoo Finance [(yfinance)](https://pypi.org/project/yfinance/) to fetch historical stock prices.

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack StockMarketDataQueryEnginePack --download-dir ./stock_market_data_pack
```

You can then inspect the files at `./stock_market_data_pack` and use them as a template for your own project.

## Code Usage

You can download the pack to a the `./stock_market_data_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
StockMarketDataQueryEnginePack = download_llama_pack(
    "StockMarketDataQueryEnginePack", "./stock_market_data_pack"
)
```

From here, you can use the pack, or inspect and modify the pack in `./stock_market_data_pack`.

Then, you can set up the pack like so:

```python
# create the pack
stock_market_data_pack = StockMarketDataQueryEnginePack(
    ["MSFT"],
    period="1mo",
)
```

The `run()` function is a light wrapper around `query_engine.query()`.

```python
response = stock_market_data_pack.run(
    "What is the average closing price for MSFT?"
)
```
