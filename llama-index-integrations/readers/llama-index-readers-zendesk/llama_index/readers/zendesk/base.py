"""Zendesk reader."""

import json
from typing import List

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class ZendeskReader(BaseReader):
    """
    Zendesk reader. Reads data from a Zendesk workspace.

    Args:
        zendesk_subdomain (str): Zendesk subdomain
        locale (str): Locale of articles

    """

    def __init__(self, zendesk_subdomain: str, locale: str = "en-us") -> None:
        """Initialize Zendesk reader."""
        self.zendesk_subdomain = zendesk_subdomain
        self.locale = locale

    def load_data(self) -> List[Document]:
        """
        Load data from the workspace.

        Args:
            workspace_id (str): Workspace ID.


        Returns:
            List[Document]: List of documents.

        """
        from bs4 import BeautifulSoup

        results = []

        articles = self.get_all_articles()
        for article in articles:
            body = article["body"]
            if body is None:
                continue
            soup = BeautifulSoup(body, "html.parser")
            body = soup.get_text()
            extra_info = {
                "id": article["id"],
                "title": article["title"],
                "url": article["html_url"],
                "updated_at": article["updated_at"],
            }

            results.append(
                Document(
                    text=body,
                    extra_info=extra_info,
                )
            )

        return results

    def get_all_articles(self):
        articles = []
        next_page = None

        while True:
            response = self.get_articles_page(next_page)
            articles.extend(response["articles"])
            next_page = response["next_page"]

            if next_page is None:
                break

        return articles

    def get_articles_page(self, next_page: str = None):
        import requests

        if next_page is None:
            url = f"https://{self.zendesk_subdomain}.zendesk.com/api/v2/help_center/{self.locale}/articles?per_page=100"
        else:
            url = next_page

        response = requests.get(url)

        response_json = json.loads(response.text)

        next_page = response_json.get("next_page", None)

        articles = response_json.get("articles", [])

        return {"articles": articles, "next_page": next_page}
