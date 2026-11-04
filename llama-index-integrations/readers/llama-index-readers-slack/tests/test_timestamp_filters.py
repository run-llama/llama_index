from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

from llama_index.readers.slack import SlackReader


@pytest.fixture()
def client():
    with patch("slack_sdk.WebClient") as client_cls:
        client = client_cls.return_value
        client.api_test.return_value = {"ok": True}
        client.conversations_history.return_value = {
            "messages": [{"ts": "0.0"}],
            "has_more": False,
        }
        client.conversations_replies.return_value = {
            "messages": [{"text": "message"}],
            "has_more": False,
        }
        yield client


@pytest.mark.parametrize("latest", [0.0, 1234.5])
def test_explicit_latest_timestamp_reaches_slack(client, latest):
    reader = SlackReader(
        slack_token="placeholder-token",
        earliest_date_timestamp=0.0,
        latest_date_timestamp=latest,
    )

    documents = reader.load_data(["channel"])

    assert reader.latest_date_timestamp == latest
    assert documents[0].text == "message"
    assert client.conversations_history.call_args.kwargs["latest"] == str(latest)
    assert client.conversations_replies.call_args.kwargs["latest"] == str(latest)


def test_latest_datetime_takes_precedence(client):
    latest = datetime(2024, 1, 2, tzinfo=timezone.utc)
    reader = SlackReader(
        slack_token="placeholder-token",
        earliest_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        latest_date=latest,
        latest_date_timestamp=1234.5,
    )
    assert reader.latest_date_timestamp == latest.timestamp()


def test_missing_latest_timestamp_defaults_to_now(client):
    with patch("llama_index.readers.slack.base.datetime") as clock:
        clock.now.return_value = Mock(timestamp=Mock(return_value=5678.9))
        reader = SlackReader(slack_token="placeholder-token")
    assert reader.latest_date_timestamp == 5678.9
