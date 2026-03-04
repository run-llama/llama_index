"""Feedly Rss Reader."""

import json
from pathlib import Path

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class FeedlyRssReader(BaseReader):
    """
    Feedly Rss Reader.

    Get entries from Feedly Rss Reader

    Uses Feedly Official python-api-client: https://github.com/feedly/python-api-client
    """

    def __init__(self, bearer_token: str) -> None:
        """Initialize with parameters."""
        super().__init__()
        self.bearer_token = bearer_token

    def setup_auth(
        self, directory: Path = Path.home() / ".config/feedly", overwrite: bool = False
    ):
        """
        Modified from python-api-client/feedly/api_client/utils.py
        Instead promopting for user input, we take the token as an argument.
        """
        directory.mkdir(exist_ok=True, parents=True)

        auth_file = directory / "access.token"

        if not auth_file.exists() or overwrite:
            auth = self.bearer_token
            auth_file.write_text(auth.strip())

    def load_data(self, category_name, max_count=100):
        """Get the entries from a feedly category."""
        from feedly.api_client.session import FeedlySession
        from feedly.api_client.stream import StreamOptions

        self.setup_auth(overwrite=True)
        sess = FeedlySession()
        category = sess.user.user_categories.get(category_name)

        documents = []
        for article in category.stream_contents(
            options=StreamOptions(max_count=max_count)
        ):
            # doc for available fields: https://developer.feedly.com/v3/streams/
            entry = {
                "title": article["title"],
                "published": article["published"],
                "summary": article["summary"],
                "author": article["author"],
                "content": article["content"],
                "keywords": article["keywords"],
                "commonTopics": article["commonTopics"],
            }

            text = json.dumps(entry, ensure_ascii=False)

            documents.append(Document(text=text))
        return documents
