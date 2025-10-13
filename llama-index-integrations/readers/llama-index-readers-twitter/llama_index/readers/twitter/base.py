"""Simple reader that reads tweets of a twitter handle."""

from typing import Any, List, Optional

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document


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
            response = " "
            for tweet in tweets.data:
                response = response + tweet.text + "\n"
            results.append(Document(text=response, id_=username))
        return results
