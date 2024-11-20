"""Wordpress reader."""
import warnings
from typing import List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class WordpressReader(BaseReader):
    """Wordpress reader. Reads data from a Wordpress workspace.

    Args:
        url (str): Base URL of the WordPress site.
        username (Optional[str]): WordPress username for authentication.
        password (Optional[str]): WordPress password for authentication.
        get_pages (bool): Retrieve static WordPress 'pages'. Default True.
        get_posts (bool): Retrieve WordPress 'posts' (blog entries). Default True.
        additional_post_types (Optional[str]): Comma-separated list of additional post types to retrieve
                                               (e.g., 'my-custom-page,webinars'). Default is None.
    """

    def __init__(
        self,
        url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        get_pages: bool = True,
        get_posts: bool = True,
        additional_post_types: Optional[str] = None,
    ) -> None:
        """Initialize Wordpress reader."""
        self.url = url
        self.username = username
        self.password = password

        # Use a set to prevent duplicates
        self.post_types = set()

        # Add default types based on backward-compatible options
        if get_pages:
            self.post_types.add("pages")
        if get_posts:
            self.post_types.add("posts")

        # Add any additional post types specified
        if additional_post_types:
            self.post_types.update(
                post_type.strip() for post_type in additional_post_types.split(",")
            )

        # Convert post_types back to a list
        self.post_types = list(self.post_types)

    def load_data(self) -> List[Document]:
        """Load data from the specified post types.

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

        # Fetch articles for each specified post type
        for post_type in self.post_types:
            articles.extend(self.get_all_posts(post_type))

        # Process each article to extract content and metadata
        for article in articles:
            body = article.get("content", {}).get("rendered", None)
            if body is None:
                body = article.get("content")

            soup = BeautifulSoup(body)
            body = soup.get_text()

            title = article.get("title", {}).get("rendered", None) or article.get(
                "title"
            )

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

    def get_all_posts(self, post_type: str) -> List[dict]:
        """Retrieve all posts of a specific type, handling pagination."""
        posts = []
        next_page = 1

        while True:
            response = self.get_posts_page(post_type, next_page)
            posts.extend(response["articles"])
            next_page = response["next_page"]

            if next_page is None:
                break

        return posts

    def get_posts_page(self, post_type: str, current_page: int = 1) -> dict:
        """Retrieve a single page of posts for a given post type."""
        import requests

        url = f"{self.url}/wp-json/wp/v2/{post_type}?per_page=100&page={current_page}"

        # Handle authentication if username and password are provided
        auth = (
            (self.username, self.password) if self.username and self.password else None
        )
        response = requests.get(url, auth=auth)
        response.raise_for_status()  # Raise an error for bad responses

        headers = response.headers
        num_pages = int(headers.get("X-WP-TotalPages", 1))
        next_page = current_page + 1 if num_pages > current_page else None

        articles = response.json()
        return {"articles": articles, "next_page": next_page}
