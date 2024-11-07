"""Wordpress reader."""
import json
import warnings

from typing import List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class WordpressReader(BaseReader):
    """Wordpress reader. Reads data from a Wordpress workspace.

    Args:
        wordpress_subdomain (str): Wordpress subdomain
        get_pages (bool): Retrieve static Wordpress 'pages'.  Default True.
        get_posts (bool): Retrieve Wordpress 'posts' (blog entries).  Default True.
    """

    def __init__(
        self,
        url: str,
        password: Optional[str] = None,
        username: Optional[str] = None,
        get_pages: bool = True,
        get_posts: bool = True,
    ) -> None:
        """Initialize Wordpress reader."""
        self.url = url
        self.username = username
        self.password = password
        self.get_pages = get_pages
        self.get_posts = get_posts

    def load_data(self) -> List[Document]:
        """Load data from the workspace.

        Returns:
            List[Document]: List of documents.
        """
        from bs4 import BeautifulSoup, GuessedAtParserWarning

        #  Suppressing this warning because guessing at the parser is the
        #  desired behavior -- we don't want to force lxml on packages
        #  where it's not installed.
        warnings.filterwarnings("ignore", category=GuessedAtParserWarning)

        results = []
        articles = []

        if self.get_pages:
            articles.extend(self.get_all_posts("pages"))

        if self.get_posts:
            articles.extend(self.get_all_posts("posts"))

        for article in articles:
            body = article.get("content", {}).get("rendered", None)
            if body is None:
                body = article.get("content")

            soup = BeautifulSoup(body)
            body = soup.get_text()

            title = article.get("title", {}).get("rendered", None)
            if not title:
                title = article.get("title")

            extra_info = {
                "id": article["id"],
                "title": title,
                "url": article["link"],
                "updated_at": article["modified"],
            }

            results.append(
                Document(
                    text=body,
                    extra_info=extra_info,
                )
            )
        return results

    def get_all_posts(self, post_type: str):
        posts = []
        next_page = 1

        while True:
            response = self.get_posts_page(post_type, next_page)
            posts.extend(response["articles"])
            next_page = response["next_page"]

            if next_page is None:
                break

        return posts

    def get_posts_page(self, post_type: str, current_page: int = 1):
        import requests

        url = f"{self.url}/wp-json/wp/v2/{post_type}?per_page=100&page={current_page}"

        response = requests.get(url)
        headers = response.headers

        if "X-WP-TotalPages" in headers:
            num_pages = int(headers["X-WP-TotalPages"])
        else:
            num_pages = 1

        if num_pages > current_page:
            next_page = current_page + 1
        else:
            next_page = None

        response_json = json.loads(response.text)

        articles = response_json

        return {"articles": articles, "next_page": next_page}
