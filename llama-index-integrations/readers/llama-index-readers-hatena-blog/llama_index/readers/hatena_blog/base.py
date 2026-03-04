"""Hatena Blog reader."""

from typing import Dict, List

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

ATOM_PUB_ENTRY_URL = "{root_endpoint}/entry"


class Article:
    def __init__(self) -> None:
        self.title = ""
        self.content = ""
        self.published = ""
        self.url = ""


class HatenaBlogReader(BaseReader):
    """
    Hatena Blog reader.

    Args:
        root_endpoint (str): AtomPub root endpoint.
        api_key (str): AtomPub API Key
        username (str): Hatena ID

    """

    def __init__(self, root_endpoint: str, api_key: str, username: str) -> None:
        """Initialize Hatena Blog reader."""
        self.root_endpoint = root_endpoint
        self.api_key = api_key
        self.username = username

    def load_data(self) -> List[Document]:
        results = []
        articles = self.get_all_articles()
        for a in articles:
            results.append(
                Document(
                    text=a.content,
                    extra_info={
                        "title": a.title,
                        "published": a.published,
                        "url": a.url,
                    },
                )
            )

        return results

    def get_all_articles(self) -> List[Article]:
        articles: List[Article] = []
        page_url = ATOM_PUB_ENTRY_URL.format(root_endpoint=self.root_endpoint)

        while True:
            res = self.get_articles(page_url)
            articles += res.get("articles")
            page_url = res.get("next_page")
            if page_url is None:
                break

        return articles

    def get_articles(self, url: str) -> Dict:
        import requests
        from bs4 import BeautifulSoup
        from requests.auth import HTTPBasicAuth

        articles: List[Article] = []
        next_page = None

        res = requests.get(url, auth=HTTPBasicAuth(self.username, self.api_key))
        soup = BeautifulSoup(res.text, "xml")
        for entry in soup.find_all("entry"):
            if entry.find("app:control").find("app:draft").string == "yes":
                continue
            article = Article()
            article.title = entry.find("title").string
            article.published = entry.find("published").string
            article.url = entry.find("link", rel="alternate")["href"]
            content = entry.find("content")
            if content.get("type") == "text/html":
                article.content = (
                    BeautifulSoup(entry.find("content").string, "html.parser")
                    .get_text()
                    .strip()
                )
            else:
                article.content = entry.find("content").string.strip()
            articles.append(article)

        next = soup.find("link", attrs={"rel": "next"})
        if next:
            next_page = next.get("href")

        return {"articles": articles, "next_page": next_page}
