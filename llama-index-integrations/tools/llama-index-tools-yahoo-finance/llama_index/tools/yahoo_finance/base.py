from llama_index.core.tools.tool_spec.base import BaseToolSpec
import yfinance as yf
import pandas as pd


class YahooFinanceToolSpec(BaseToolSpec):
    """Yahoo Finance tool spec."""

    spec_functions = [
        "balance_sheet",
        "income_statement",
        "cash_flow",
        "stock_basic_info",
        "stock_analyst_recommendations",
        "stock_news",
    ]

    def __init__(self) -> None:
        """Initialize the Yahoo Finance tool spec."""

    def balance_sheet(self, ticker: str) -> str:
        """
        Return the balance sheet of the stock.

        Args:
          ticker (str): the stock ticker to be given to yfinance

        """
        stock = yf.Ticker(ticker)
        balance_sheet = stock.balance_sheet
        if balance_sheet is None or balance_sheet.empty:
            return "Balance sheet data is not available."
        return "Balance Sheet: \n" + pd.DataFrame(balance_sheet).to_string()

    def income_statement(self, ticker: str) -> str:
        """
        Return the income statement of the stock.

        Args:
          ticker (str): the stock ticker to be given to yfinance

        """
        stock = yf.Ticker(ticker)
        income_stmt = stock.income_stmt
        if income_stmt is None or income_stmt.empty:
            return "Income statement data is not available."
        return "Income Statement: \n" + pd.DataFrame(income_stmt).to_string()

    def cash_flow(self, ticker: str) -> str:
        """
        Return the cash flow of the stock.

        Args:
          ticker (str): the stock ticker to be given to yfinance

        """
        stock = yf.Ticker(ticker)
        cash_flow = stock.cashflow
        if cash_flow is None or cash_flow.empty:
            return "Cash flow data is not available."
        return "Cash Flow: \n" + pd.DataFrame(cash_flow).to_string()

    def stock_basic_info(self, ticker: str) -> str:
        """
        Return the basic info of the stock. Ex: price, description, name.

        Args:
          ticker (str): the stock ticker to be given to yfinance

        """
        stock = yf.Ticker(ticker)
        return "Info: \n" + str(stock.info)

    def stock_analyst_recommendations(self, ticker: str) -> str:
        """
        Get the analyst recommendations for a stock.

        Args:
          ticker (str): the stock ticker to be given to yfinance

        """
        stock = yf.Ticker(ticker)
        return "Recommendations: \n" + str(stock.recommendations)

    def stock_news(self, ticker: str) -> str:
        """
        Get the most recent news titles of a stock.

        Args:
          ticker (str): the stock ticker to be given to yfinance

        """
        stock = yf.Ticker(ticker)
        news = stock.news
        if news is None or len(news) == 0:
            return "No news available."
        out = "News: \n"
        for i in news:
            out += i.get("title", "No title") + "\n"
        return out
