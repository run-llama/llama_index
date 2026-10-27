from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.finance import FinanceAgentToolSpec


def test_class():
    name_of_base_classes = [b.__name__ for b in FinanceAgentToolSpec.__mro__]
    assert BaseToolSpec.__name__ in name_of_base_classes


def test_get_upcoming_earnings_sends_only_alpha_vantage_key():
    from unittest.mock import MagicMock, patch

    from llama_index.tools.finance import earnings

    csv_body = (
        "symbol,name,reportDate,fiscalDateEnding,estimate,currency\n"
        "AAA,Alpha Inc,2030-01-10,2029-12-31,1.0,USD\n"
        "BBB,Beta Inc,2030-01-11,2029-12-31,1.0,EUR\n"
        "CCC,Gamma Inc,2031-01-10,2030-12-31,1.0,USD\n"
    )
    response = MagicMock()
    response.content = csv_body.encode("utf-8")
    session = MagicMock()
    session.__enter__.return_value = session
    session.get.return_value = response

    api_key = {
        "ALPHA_VANTAGE": "av-key",
        "POLYGON": "polygon-key",
        "FINNHUB": "finnhub-key",
        "NEWSAPI": "newsapi-key",
    }
    with patch.object(earnings.requests, "Session", return_value=session):
        df = earnings.get_upcoming_earnings(
            api_key,
            start_date="2030-01-01",
            end_date="2030-02-01",
            country="USD",
            only_sp500=False,
        )

    url = session.get.call_args.args[0]
    assert url.endswith("&apikey=av-key")
    assert "polygon-key" not in url
    assert list(df["symbol"]) == ["AAA"]
