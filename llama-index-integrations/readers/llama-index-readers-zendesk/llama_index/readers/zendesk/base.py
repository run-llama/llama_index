"""Zendesk reader."""

import json
from typing import Any, Dict, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class ZendeskReader(BaseReader):
    """
    Zendesk reader. Reads data from a Zendesk workspace.

    Args:
        zendesk_subdomain (str): Zendesk subdomain
        locale (str): Locale of articles
        timeout (Optional[float]): Request timeout in seconds.

    """

    def __init__(
        self,
        zendesk_subdomain: str,
        locale: str = "en-us",
        timeout: Optional[float] = None,
    ) -> None:
        """Initialize Zendesk reader."""
        self.zendesk_subdomain = zendesk_subdomain
        self.locale = locale
        self.timeout = timeout

    def load_data(self) -> List[Document]:
        """
        Load data from the workspace.

        Returns:
            List[Document]: List of documents.

        """
        from bs4 import BeautifulSoup

        results = []

        articles = self.get_all_articles()
        for article in articles:
            body = article.get("body")
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
                    metadata=extra_info,
                )
            )

        return results

    def get_all_articles(self) -> List[Dict[str, Any]]:
        articles = []
        next_page = None

        while True:
            response = self.get_articles_page(next_page)
            articles.extend(response["articles"])
            next_page = response["next_page"]

            if next_page is None:
                break

        return articles

    def get_articles_page(
        self,
        next_page: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        import requests

        if next_page is None:
            url = f"https://{self.zendesk_subdomain}.zendesk.com/api/v2/help_center/{self.locale}/articles?per_page=100"
        else:
            url = next_page

        request_timeout = timeout if timeout is not None else self.timeout
        response = requests.get(url, timeout=request_timeout)
        response.raise_for_status()

        response_json = json.loads(response.text)
        if not isinstance(response_json, dict):
            raise ValueError(
                f"Expected JSON object in Zendesk response, got {type(response_json).__name__}"
            )

        next_page = response_json.get("next_page", None)
        articles = response_json.get("articles", [])

        return {"articles": articles, "next_page": next_page}
