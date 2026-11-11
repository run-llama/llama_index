from email.message import Message
from io import BytesIO
from urllib.request import OpenerDirector
from urllib.response import addinfourl

import feedparser
import pytest

from llama_index.readers.web import RssReader


@pytest.fixture()
def sent_agents(monkeypatch):
    agents = []
    monkeypatch.setattr(feedparser, "USER_AGENT", "ApplicationDefault/1.0")

    def open_response(self, request):
        agents.append(request.get_header("User-agent"))
        headers = Message()
        headers["Content-Type"] = "application/rss+xml; charset=utf-8"
        body = b"""<rss version="2.0"><channel><title>Test feed</title>
        <item><title>Entry</title><guid>entry-1</guid>
        <description>Feed content</description></item></channel></rss>"""
        return addinfourl(BytesIO(body), headers, request.full_url, code=200)

    monkeypatch.setattr(OpenerDirector, "open", open_response)
    return agents


def test_custom_agent_does_not_change_feedparser_default(sent_agents):
    documents = RssReader(user_agent="ReaderA/1.0").load_data(
        ["https://example.test/feed"]
    )

    assert sent_agents == ["ReaderA/1.0"]
    assert documents[0].text == "Feed content"
    assert documents[0].metadata["title"] == "Entry"
    assert feedparser.USER_AGENT == "ApplicationDefault/1.0"


def test_default_reader_keeps_application_agent_after_custom_reader(sent_agents):
    RssReader(user_agent="ReaderA/1.0").load_data(
        ["https://example.test/first", "https://example.test/second"]
    )
    RssReader().load_data(["https://example.test/default"])

    assert sent_agents == ["ReaderA/1.0", "ReaderA/1.0", "ApplicationDefault/1.0"]


def test_custom_readers_keep_their_own_agents(sent_agents):
    first = RssReader(user_agent="ReaderA/1.0")
    second = RssReader(user_agent="ReaderB/1.0")
    for reader in (first, second, first):
        reader.load_data(["https://example.test/feed"])

    assert sent_agents == ["ReaderA/1.0", "ReaderB/1.0", "ReaderA/1.0"]


def test_empty_url_list_does_not_change_application_agent(sent_agents):
    documents = RssReader(user_agent="ReaderA/1.0").load_data([])

    assert documents == []
    assert sent_agents == []
    assert feedparser.USER_AGENT == "ApplicationDefault/1.0"
