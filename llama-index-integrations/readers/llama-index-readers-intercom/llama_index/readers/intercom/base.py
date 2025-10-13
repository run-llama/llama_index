"""Intercom reader."""

import json
from typing import List

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class IntercomReader(BaseReader):
    """
    Intercom reader. Reads data from a Intercom workspace.

    Args:
        personal_access_token (str): Intercom token.

    """

    def __init__(self, intercom_access_token: str) -> None:
        """Initialize Intercom reader."""
        self.intercom_access_token = intercom_access_token

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
            soup = BeautifulSoup(body, "html.parser")
            body = soup.get_text()

            extra_info = {
                "id": article["id"],
                "title": article["title"],
                "url": article["url"],
                "updated_at": article["updated_at"],
            }

            results.append(
                Document(
                    text=body,
                    extra_info=extra_info or {},
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
            url = "https://api.intercom.io/articles"
        else:
            url = next_page

        headers = {
            "accept": "application/json",
            "Intercom-Version": "2.8",
            "authorization": f"Bearer {self.intercom_access_token}",
        }

        response = requests.get(url, headers=headers)

        response_json = json.loads(response.text)

        next_page = response_json.get("pages", {}).get("next", None)

        articles = response_json.get("data", [])

        return {"articles": articles, "next_page": next_page}
