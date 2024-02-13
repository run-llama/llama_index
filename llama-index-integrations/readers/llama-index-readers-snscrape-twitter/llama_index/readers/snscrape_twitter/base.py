"""SnscrapeTwitter reader."""
from typing import List

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class SnscrapeTwitterReader(BaseReader):
    """SnscrapeTwitter reader. Reads data from a twitter profile.

    Args:
        username (str): Twitter Username.
        num_tweets (int): Number of tweets to fetch.
    """

    def __init__(self) -> None:
        """Initialize SnscrapeTwitter reader."""

    def load_data(self, username: str, num_tweets: int) -> List[Document]:
        """Load data from a twitter profile.

        Args:
            username (str): Twitter Username.
            num_tweets (int): Number of tweets to fetch.


        Returns:
            List[Document]: List of documents.
        """
        import snscrape.modules.twitter as sntwitter

        attributes_container = []
        for i, tweet in enumerate(
            sntwitter.TwitterSearchScraper(f"from:{username}").get_items()
        ):
            if i > num_tweets:
                break
            attributes_container.append(tweet.rawContent)
        return [Document(text=attributes_container, extra_info={"username": username})]
