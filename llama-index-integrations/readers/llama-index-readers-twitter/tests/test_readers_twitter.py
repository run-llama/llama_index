import sys
from types import SimpleNamespace

import pytest
from llama_index.core.readers.base import BasePydanticReader
from llama_index.readers.twitter import TwitterTweetReader, XquikTweetReader


def test_class():
    names_of_base_classes = [b.__name__ for b in TwitterTweetReader.__mro__]
    assert BasePydanticReader.__name__ in names_of_base_classes


def test_twitter_reader_handles_empty_timeline(monkeypatch):
    client = SimpleNamespace(
        get_user=lambda **kwargs: SimpleNamespace(data=SimpleNamespace(id="42")),
        get_users_tweets=lambda *args, **kwargs: SimpleNamespace(data=None),
    )
    tweepy = SimpleNamespace(Client=lambda **kwargs: client)
    monkeypatch.setitem(sys.modules, "tweepy", tweepy)

    documents = TwitterTweetReader(bearer_token="test-token").load_data(["empty_user"])

    assert len(documents) == 1
    assert documents[0].id_ == "empty_user"
    assert documents[0].text == ""


class FakeXquikUsers:
    def __init__(self, tweets):
        self.tweets = tweets
        self.calls = []

    def retrieve_tweets(self, username, **kwargs):
        self.calls.append((username, kwargs))
        return SimpleNamespace(tweets=self.tweets)


def make_xquik_client(tweets):
    users = FakeXquikUsers(tweets)
    return SimpleNamespace(x=SimpleNamespace(users=users)), users


def test_xquik_reader_loads_bounded_timeline_and_hides_key():
    client, users = make_xquik_client(
        [SimpleNamespace(text="First post"), SimpleNamespace(text="Second post")]
    )
    api_key = "xq_test_secret_value"
    reader = XquikTweetReader(api_key=api_key, num_tweets=25, client=client)

    documents = reader.load_data(["@llama_index"], num_tweets=2)

    assert users.calls == [
        (
            "llama_index",
            {
                "page_size": 2,
                "include_replies": False,
                "include_parent_tweet": False,
            },
        )
    ]
    assert len(documents) == 1
    assert documents[0].id_ == "llama_index"
    assert documents[0].text == "First post\nSecond post"
    assert documents[0].metadata == {
        "source": "xquik",
        "username": "llama_index",
        "tweet_count": 2,
    }
    assert api_key not in repr(reader)
    assert api_key not in str(reader.model_dump())


def test_xquik_reader_preserves_empty_timeline():
    client, _ = make_xquik_client([])

    documents = XquikTweetReader(api_key="xq_test", client=client).load_data(
        ["empty_user"]
    )

    assert documents[0].text == ""
    assert documents[0].metadata["tweet_count"] == 0


@pytest.mark.parametrize("handle", ["", "@", "bad-handle", "a" * 16])
def test_xquik_reader_rejects_invalid_handles(handle):
    client, users = make_xquik_client([])
    reader = XquikTweetReader(api_key="xq_test", client=client)

    with pytest.raises(ValueError, match="Invalid X handle"):
        reader.load_data([handle])

    assert users.calls == []


@pytest.mark.parametrize("num_tweets", [0, 301, True])
def test_xquik_reader_rejects_invalid_tweet_counts(num_tweets):
    client, _ = make_xquik_client([])

    with pytest.raises(ValueError, match="between 1 and 300"):
        XquikTweetReader(api_key="xq_test", num_tweets=num_tweets, client=client)


def test_xquik_reader_rejects_malformed_response():
    client, _ = make_xquik_client(None)
    reader = XquikTweetReader(api_key="xq_test", client=client)

    with pytest.raises(ValueError, match="invalid tweets page"):
        reader.load_data(["llama_index"])


def test_xquik_reader_reports_missing_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "x_twitter_scraper", None)

    with pytest.raises(ImportError, match="readers-twitter\\[xquik\\]"):
        XquikTweetReader(api_key="xq_test")
