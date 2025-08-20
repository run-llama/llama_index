from typing import List, Optional, Any, Dict

import pandas as pd
from datetime import datetime, timedelta
from llama_index.tools.finance import comparisons, earnings, news
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class FinanceAgentToolSpec(BaseToolSpec):
    spec_functions = [
        "find_similar_companies",
        "get_earnings_history",
        "get_stocks_with_upcoming_earnings",
        "get_current_gainer_stocks",
        "get_current_loser_stocks",
        "get_current_undervalued_growth_stocks",
        "get_current_technology_growth_stocks",
        "get_current_most_traded_stocks",
        "get_current_undervalued_large_cap_stocks",
        "get_current_aggressive_small_cap_stocks",
        "get_trending_finance_news",
        "get_google_trending_searches",
        "get_google_trends_for_query",
        "get_latest_news_for_stock",
        "get_current_stock_price_info",
    ]

    def __init__(
        self,
        polygon_api_key: str,
        finnhub_api_key: str,
        alpha_vantage_api_key: str,
        newsapi_api_key: str,
    ):
        self._api_key = {
            "ALPHA_VANTAGE": alpha_vantage_api_key,
            "POLYGON": polygon_api_key,
            "FINNHUB": finnhub_api_key,
            "NEWSAPI": newsapi_api_key,
        }

    def find_similar_companies(self, symbol: str) -> List[str]:
        """Given a stock's ticker symbol, returns a list of similar companies."""
        return comparisons.find_similar_companies(self._api_key, symbol)

    def get_earnings_history(self, symbol: str) -> pd.DataFrame:
        """Given a stock's ticker symbol, returns a dataframe storing actual and estimated earnings over past K quarterly reports."""
        return earnings.get_earnings_history(self._api_key, symbol)

    def get_latest_earning_estimate(self, symbol: str) -> float:
        """Given a stock's ticker symbol, returns it's earnings estimate for the upcoming quarterly report."""
        return earnings.get_latest_earning_estimate(symbol)

    def get_stocks_with_upcoming_earnings(
        self, num_days_from_now: int, only_sp500: bool
    ) -> pd.DataFrame:
        """
        Returns a pandas dataframe containing all stocks which are announcing earnings in upcoming days.

        Arguments:
         num_days_from_now: only returns stocks which announcing earnings from today's date to num_days_from_now.
         only_sp500: only returns sp500 stocks.

        """
        start_date = datetime.now().strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(num_days_from_now)).strftime("%Y-%m-%d")
        return earnings.get_upcoming_earnings(
            self._api_key,
            start_date=start_date,
            end_date=end_date,
            country="USD",
            only_sp500=only_sp500,
        )

    def get_current_gainer_stocks(self) -> pd.DataFrame:
        """
        Return US stocks which are classified as day gainers as per Yahoo Finance.

        A US stock is classified as day gainer if %change in price > 3, price >=5, volume > 15_000

        """
        return news.get_current_gainer_stocks()

    def get_current_loser_stocks(self) -> pd.DataFrame:
        """
        Returns US stocks which are classified as day losers as per Yahoo Finance.

        A US stock is classified as day loser if %change in price < -2.5, price >=5, volume > 20_000

        """
        return news.get_current_loser_stocks()

    def get_current_undervalued_growth_stocks(self) -> pd.DataFrame:
        """
        Get list of undervalued growth stocks in US market as per Yahoo Finance.

        A stock with Price to Earnings ratio between 0-20, Price / Earnings to Growth < 1

        """
        return news.get_current_undervalued_growth_stocks()

    def get_current_technology_growth_stocks(self) -> pd.DataFrame:
        """
        Returns a data frame of growth stocks in technology sector in US market.

        If a stocks's quarterly revenue growth YoY% > 25%.

        """
        return news.get_current_technology_growth_stocks()

    def get_current_most_traded_stocks(self) -> pd.DataFrame:
        """
        Returns a dataframe storing stocks which were traded the most in current market.

        Stocks are ordered in decreasing order of activity i.e stock traded the most on top.

        """
        return news.get_current_most_traded_stocks()

    def get_current_undervalued_large_cap_stocks(self) -> pd.DataFrame:
        """Returns a dataframe storing US market large cap stocks with P/E < 20."""
        return news.get_current_undervalued_large_cap_stocks()

    def get_current_aggressive_small_cap_stocks(self) -> pd.DataFrame:
        """Returns a dataframe storing US market small cap stocks with 1 yr % change in earnings per share > 25."""
        return news.get_current_aggressive_small_cap_stocks()

    def get_trending_finance_news(self) -> List[str]:
        """Returns a list of top 10 trending news in financial market as per seekingalpha."""
        trends = news.get_topk_trending_news()
        return [t["title"] for t in trends]

    def get_google_trending_searches(self) -> Optional[pd.DataFrame]:
        """
        Returns trending searches in US as per google trends.

        If unable to find any trends, returns None.

        """
        return news.get_google_trending_searches(region="united_states")

    def get_google_trends_for_query(self, query: str) -> Optional[pd.DataFrame]:
        """
        Finds google search trends for a given query in United States.

        Returns None if unable to find any trends.

        """
        return news.get_google_trends_for_query(query=query, region="united_states")

    def get_latest_news_for_stock(self, stock_id: str) -> List[str]:
        """Given a stock_id representing the name of a company or the stock ticker symbol, Returns a list of news published related to top business articles in US in last 7 days from now."""
        articles = news.get_latest_news_for_stock(self._api_key, stock_id=stock_id)
        return [a["title"] for a in articles]

    def get_current_stock_price_info(
        self, stock_ticker_symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        Given a stock's ticker symbol, returns current price information of the stock.

        Returns None if the provided stock ticker symbol is invalid.
        """
        price_info = news.get_current_stock_price_info(
            self._api_key, stock_ticker_symbol
        )
        if price_info is not None:
            return {
                "Current Price": price_info["c"],
                "High Price of the day": price_info["h"],
                "Low Price of the day": price_info["l"],
                "Open Price of the day": price_info["o"],
                "Percentage change": price_info["dp"],
            }
        return None
