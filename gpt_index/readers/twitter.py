"""Simple reader that reads tweets of a twitter handle."""
from typing import Any, List, Optional

from gpt_index.readers.base import BaseReader
from gpt_index.readers.schema.base import Document


class TwitterTweetReader(BaseReader):
    """Twitter tweets reader.

    Read tweets of user twitter handle.

    Check 'https://developer.twitter.com/en/docs/twitter-api/\
        getting-started/getting-access-to-the-twitter-api' \
        on how to get access to twitter API.

    Args:
        consumer_key (str): consumer_key that you get from twitter API.
        consumer_secret (str): consumer_secret that you get from twitter API.
        access_token (str): access_token that you get from twitter API.
        access_token_secret (str): access_token_secret that you get from twitter API.
        num_tweets (Optional[int]): Number of tweets for each user twitter handle.\
            Default is 100 tweets.
    """

    def __init__(
        self,
        consumer_key: str,
        consumer_secret: str,
        access_token: str,
        access_token_secret: str,
        num_tweets: Optional[int] = 100,
        verbose: bool = False,
    ) -> None:
        """Initialize with parameters."""
        super().__init__(verbose=verbose)
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret
        self.num_tweets = num_tweets

    def load_data(
        self, twitterhandles: List[str], **load_kwargs: Any
    ) -> List[Document]:
        """Load tweets of twitter handles.

        Args:
            twitterhandles (List[str]): List of user twitter handles to read tweets.

        """
        try:
            import tweepy
        except ImportError:
            raise ValueError(
                "`tweepy` package not found, please run `pip install tweepy`"
            )

        # Authenticate with Twitter API
        auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
        auth.set_access_token(self.access_token, self.access_token_secret)
        api = tweepy.API(auth)

        results = []
        for user in twitterhandles:
            tweets = api.user_timeline(screen_name=user, count=self.num_tweets)
            response = " "
            for tweet in tweets:
                response = response + tweet.text + "\n"
            results.append(Document(response))
        return results
