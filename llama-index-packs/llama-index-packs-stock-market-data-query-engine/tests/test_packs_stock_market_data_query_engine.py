from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.packs.stock_market_data_query_engine import (
    StockMarketDataQueryEnginePack,
)


def test_class():
    names_of_base_classes = [b.__name__ for b in StockMarketDataQueryEnginePack.__mro__]
    assert BaseLlamaPack.__name__ in names_of_base_classes
