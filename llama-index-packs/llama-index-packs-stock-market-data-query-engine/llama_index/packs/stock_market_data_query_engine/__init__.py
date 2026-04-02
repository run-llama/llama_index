import warnings

warnings.warn(
    "llama-index-packs-stock-market-data-query-engine is deprecated and no longer maintained. "
    "It will not receive any further updates.",
    DeprecationWarning,
    stacklevel=2,
)

from llama_index.packs.stock_market_data_query_engine.base import (
    StockMarketDataQueryEnginePack,
)

__all__ = ["StockMarketDataQueryEnginePack"]
