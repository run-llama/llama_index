import pandas as pd

from collections import defaultdict
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from pytrends.request import TrendReq
from typing import List, Dict, Optional, Any
from newsapi import NewsApiClient

from llama_index.tools.finance.util import get_df, request


def get_current_gainer_stocks() -> pd.DataFrame:
    """Return gainers of the day from yahoo finace including all cap stocks."""
    df_gainers = get_df("https://finance.yahoo.com/screener/predefined/day_gainers")[0]
    df_gainers.dropna(how="all", axis=1, inplace=True)
    return df_gainers.replace(float("NaN"), "")


def get_current_loser_stocks() -> pd.DataFrame:
    """Get data for today's losers from yahoo finance including all cap stocks."""
    df_losers = get_df("https://finance.yahoo.com/screener/predefined/day_losers")[0]
    df_losers.dropna(how="all", axis=1, inplace=True)
    return df_losers.replace(float("NaN"), "")


def get_current_undervalued_growth_stocks() -> pd.DataFrame:
    """Get data for today's stocks with low PR ratio and growth rate better than 25%."""
    df = get_df(
        "https://finance.yahoo.com/screener/predefined/undervalued_growth_stocks"
    )[0]
    df.dropna(how="all", axis=1, inplace=True)
    return df.replace(float("NaN"), "")


def get_current_technology_growth_stocks() -> pd.DataFrame:
    """Get data for today's stocks with low PR ratio and growth rate better than 25%."""
    df = get_df(
        "https://finance.yahoo.com/screener/predefined/growth_technology_stocks"
    )[0]
    df.dropna(how="all", axis=1, inplace=True)
    return df.replace(float("NaN"), "")


def get_current_most_traded_stocks() -> pd.DataFrame:
    """Get data for today's stocks in descending order based on intraday trading volume."""
    df = get_df("https://finance.yahoo.com/screener/predefined/most_active")[0]
    df.dropna(how="all", axis=1, inplace=True)
    return df.replace(float("NaN"), "")


def get_current_undervalued_large_cap_stocks() -> pd.DataFrame:
    """Get data for today's potentially undervalued large cap stocks from Yahoo finance."""
    df = get_df("https://finance.yahoo.com/screener/predefined/undervalued_large_caps")[
        0
    ]
    df.dropna(how="all", axis=1, inplace=True)
    return df.replace(float("NaN"), "")


def get_current_aggressive_small_cap_stocks() -> pd.DataFrame:
    """Get data for today'sagressive / high growth small cap stocks from Yahoo finance."""
    df = get_df("https://finance.yahoo.com/screener/predefined/aggressive_small_caps")[
        0
    ]
    df.dropna(how="all", axis=1, inplace=True)
    return df.replace(float("NaN"), "")


def get_current_hot_penny_stocks() -> pd.DataFrame:
    """Return data for today's hot penny stocks from pennystockflow.com."""
    df = get_df("https://www.pennystockflow.com", 0)[1]
    return df.drop([10])


def get_current_stock_price_info(
    api_key: Dict[str, str], stock_ticker: str
) -> Optional[Dict[str, Any]]:
    """
    Return current price information given a stock ticker symbol.
    """
    result = request(
        f"https://finnhub.io/api/v1/quote?symbol={stock_ticker}&token={api_key['FINNHUB']}"
    )
    if result.status_code != 200:
        return None
    return result.json()


def get_latest_news_for_stock(
    api_key: Dict[str, str], stock_id: str, limit: int = 10
) -> List[Dict[str, Any]]:
    """Returns latest news for a given stock_name by querying results via newsapi."""
    newsapi = NewsApiClient(api_key=api_key["NEWSAPI"])
    cat_to_id = defaultdict(list)
    for source in newsapi.get_sources()["sources"]:
        cat_to_id[source["category"]].append(source["id"])
    business_sources = [
        "bloomberg",
        "business-insider",
        "financial-post",
        "fortune",
        "info-money",
        "the-wall-street-journal",
    ]
    for source in business_sources:
        assert source in cat_to_id["business"]

    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    articles = newsapi.get_everything(
        q=stock_id,
        sources=",".join(business_sources),
        from_param=start_date,
        to=end_date,
        language="en",
        sort_by="relevancy",
        page=1,
    )["articles"]
    return articles[:limit]


def get_topk_trending_news(
    limit: int = 10, extract_content: bool = False
) -> List[Dict[str, Any]]:
    """Returns top Kk trending news from seekingalpha."""
    articles = []
    URL = "https://seekingalpha.com/news/trending_news"
    response = request(URL)
    if response.status_code == 200:
        for item in response.json():
            article_url = item["uri"]
            if not article_url.startswith("/news/"):
                continue

            article_id = article_url.split("/")[2].split("-")[0]

            content = ""
            if extract_content:
                article_url = f"https://seekingalpha.com/api/v3/news/{article_id}"
                article_response = request(article_url)
                jdata = article_response.json()
                try:
                    content = jdata["data"]["attributes"]["content"].replace(
                        "</li>", "</li>\n"
                    )
                    content = BeautifulSoup(content, features="html.parser").get_text()
                except Exception as e:
                    print(f"Unable to extract content for: {article_url}")

            articles.append(
                {
                    "title": item["title"],
                    "publishedAt": item["publish_on"][: item["publish_on"].rfind(".")],
                    "url": "https://seekingalpha.com" + article_url,
                    "id": article_id,
                    "content": content,
                }
            )

            if len(articles) > limit:
                break

    return articles[:limit]


def get_google_trending_searches(region: str = "") -> Optional[pd.DataFrame]:
    """Returns overall trending searches in US unless region is provided."""
    # TODO(ishan): Can we filter by category?
    try:
        pytrend = TrendReq()
        return pytrend.trending_searches(pn=region if region else "united_states")
    except Exception as e:
        print(f"Unable to find google trending searches, error: {e}")
        return None


def get_google_trends_for_query(
    query: str, find_related: bool = False, region: str = ""
) -> Optional[pd.DataFrame]:
    """Find google search trends for a given query filtered by region if provided."""
    try:
        pytrend = TrendReq()
        # 12 is the category for Business and Industrial which covers most of the related
        # topics related to fin-gpt
        # Ref: https://github.com/pat310/google-trends-api/wiki/Google-Trends-Categories

        # Only search for last 30 days from now
        pytrend.build_payload(
            kw_list=[query], timeframe="today 1-m", geo=region, cat=12
        )
        return pytrend.interest_over_time()
    except Exception as e:
        print(f"Unable to find google trend for {query}, error: {e}")
        return None
