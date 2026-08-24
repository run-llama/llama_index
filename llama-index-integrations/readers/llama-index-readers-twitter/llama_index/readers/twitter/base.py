"""Readers that load posts from an X handle."""

import re
from typing import Any, List, Optional

from llama_index.core.bridge.pydantic import PrivateAttr, SecretStr
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document


_X_HANDLE_PATTERN = re.compile(r"[A-Za-z0-9_]{1,15}")


def _normalize_x_handle(handle: str) -> str:
    normalized = handle.removeprefix("@")
    if _X_HANDLE_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"Invalid X handle: {handle!r}")
    return normalized


def _validate_xquik_tweet_count(value: int) -> int:
    if isinstance(value, bool) or not 1 <= value <= 300:
        raise ValueError("num_tweets must be between 1 and 300")
    return value


def _build_xquik_client(api_key: str) -> Any:
    try:
        from x_twitter_scraper import XTwitterScraper
    except ImportError as error:
        raise ImportError(
            "`x_twitter_scraper` package not found. "
            "Install `llama-index-readers-twitter[xquik]`."
        ) from error

    return XTwitterScraper(api_key=api_key, max_retries=2, timeout=30.0)


class TwitterTweetReader(BasePydanticReader):
    """
    Twitter tweets reader.

    Read tweets of user twitter handle.

    Check 'https://developer.twitter.com/en/docs/twitter-api/\
        getting-started/getting-access-to-the-twitter-api' \
        on how to get access to twitter API.

    Args:
        bearer_token (str): bearer_token that you get from twitter API.
        num_tweets (Optional[int]): Number of tweets for each user twitter handle.\
            Default is 100 tweets.

    """

    is_remote: bool = True
    bearer_token: str
    num_tweets: Optional[int]

    def __init__(
        self,
        bearer_token: str,
        num_tweets: Optional[int] = 100,
    ) -> None:
        """Initialize with parameters."""
        super().__init__(
            num_tweets=num_tweets,
            bearer_token=bearer_token,
        )

    @classmethod
    def class_name(cls) -> str:
        return "TwitterTweetReader"

    def load_data(
        self,
        twitterhandles: List[str],
        num_tweets: Optional[int] = None,
        **load_kwargs: Any,
    ) -> List[Document]:
        """
        Load tweets of twitter handles.

        Args:
            twitterhandles (List[str]): List of user twitter handles to read tweets.

        """
        try:
            import tweepy
        except ImportError:
            raise ImportError(
                "`tweepy` package not found, please run `pip install tweepy`"
            )

        client = tweepy.Client(bearer_token=self.bearer_token)
        results = []
        for username in twitterhandles:
            # tweets = api.user_timeline(screen_name=user, count=self.num_tweets)
            user = client.get_user(username=username)
            tweets = client.get_users_tweets(
                user.data.id, max_results=num_tweets or self.num_tweets
            )
            response = "\n".join(tweet.text for tweet in (tweets.data or []))
            results.append(Document(text=response, id_=username))
        return results


class XquikTweetReader(BasePydanticReader):
    """
    Load recent X posts through the Xquik Python SDK.

    Args:
        api_key: Xquik API key.
        num_tweets: Maximum posts to load for each handle. Defaults to 100.
        client: Optional Xquik client override for testing.

    """

    is_remote: bool = True
    api_key: SecretStr
    num_tweets: int
    _client: Any = PrivateAttr()

    def __init__(
        self,
        api_key: str,
        num_tweets: int = 100,
        client: Any = None,
    ) -> None:
        """Initialize the reader."""
        if not api_key:
            raise ValueError("api_key must not be empty")
        validated_count = _validate_xquik_tweet_count(num_tweets)
        super().__init__(api_key=api_key, num_tweets=validated_count)
        self._client = client if client is not None else _build_xquik_client(api_key)

    @classmethod
    def class_name(cls) -> str:
        return "XquikTweetReader"

    def load_data(
        self,
        twitterhandles: List[str],
        num_tweets: Optional[int] = None,
        **load_kwargs: Any,
    ) -> List[Document]:
        """Load one bounded page of recent posts for each X handle."""
        count = _validate_xquik_tweet_count(
            self.num_tweets if num_tweets is None else num_tweets
        )
        results: List[Document] = []

        for handle in twitterhandles:
            username = _normalize_x_handle(handle)
            page = self._client.x.users.retrieve_tweets(
                username,
                page_size=count,
                include_replies=False,
                include_parent_tweet=False,
            )
            tweets = getattr(page, "tweets", None)
            if not isinstance(tweets, list):
                raise ValueError("Xquik returned an invalid tweets page")

            texts: List[str] = []
            for tweet in tweets:
                text = getattr(tweet, "text", None)
                if not isinstance(text, str):
                    raise ValueError("Xquik returned a tweet without text")
                texts.append(text)

            results.append(
                Document(
                    text="\n".join(texts),
                    id_=username,
                    metadata={
                        "source": "xquik",
                        "username": username,
                        "tweet_count": len(texts),
                    },
                )
            )

        return results
