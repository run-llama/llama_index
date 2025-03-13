"""Test the Slack reader."""
import os
from typing import Dict, Any
import pytest
from slack_sdk.errors import SlackApiError
from llama_index.core.readers.base import BasePydanticReader
from llama_index.readers.slack import SlackReader


def test_class():
    names_of_base_classes = [b.__name__ for b in SlackReader.__mro__]
    assert BasePydanticReader.__name__ in names_of_base_classes

@pytest.fixture()
def mock_messages() -> Dict[str, Any]:
    """Mock Slack messages response."""
    return {
        "messages": [
            {
                "text": "You have time to hop on a quick call?",
                "ts": "1234567890.123456",
                "user": "U123",
                "channel": "C123",
            },
            {
                "text": "Hello, world!",
                "ts": "1234567890.123457",
                "user": "U124",
                "bot_id": "B123",
                "channel": "C123",
            },
            {
                "text": "",  # Empty message
                "ts": "1234567890.123458",
                "user": "U125",
                "channel": "C123",
            },
        ],
        "has_more": False,
    }


def test_slack_reader_init():
    """Test SlackReader initialization."""
    # Test with direct token
    reader = SlackReader(slack_token="test-token")
    assert reader.slack_token == "test-token"

    # Test with env variable
    os.environ["SLACK_TOKEN"] = "xoxb-env-token"
    reader = SlackReader()
    assert reader.slack_token == "xoxb-env-token"

    # Test with no token
    del os.environ["SLACK_TOKEN"]
    with pytest.raises(ValueError):
        SlackReader()


def test_process_message():
    """Test message processing."""
    reader = SlackReader(slack_token="test-token")

    # Test normal message
    msg = {"text": "Hello", "ts": "123", "user": "U123", "channel": "C123"}
    doc = reader._process_message(msg, include_bots=True)
    assert doc is not None
    assert doc.text == "Hello"
    assert doc.metadata["message_id"] == "123"

    # Test bot message with include_bots=False
    bot_msg = {
        "text": "Bot Hello",
        "ts": "124",
        "user": "U124",
        "bot_id": "B123",
        "channel": "C123",
    }
    doc = reader._process_message(bot_msg, include_bots=False)
    assert doc is None

    # Test bot message with include_bots=True
    doc = reader._process_message(bot_msg, include_bots=True)
    assert doc is not None
    assert doc.text == "Bot Hello"

    # Test empty message
    empty_msg = {"text": "", "ts": "125", "user": "U125", "channel": "C123"}
    doc = reader._process_message(empty_msg, include_bots=False)
    assert doc is None


def test_get_channel_messages(mocker):
    """Test channel message fetching."""
    reader = SlackReader(slack_token="test-token")
    mock_client = mocker.Mock()

    # Mock successful response
    mock_client.conversations_history.return_value = {
        "messages": [
            {"text": "Hello", "ts": "123", "user": "U123"},
            {"text": "World", "ts": "124", "user": "U124"},
        ],
        "has_more": False,
    }

    messages = reader._get_channel_messages(mock_client, "C123", limit=10)
    assert len(messages) == 2
    assert messages[0]["text"] == "Hello"

    # Test with thread replies
    mock_client.conversations_history.return_value = {
        "messages": [
            {"text": "Thread parent", "ts": "123", "user": "U123", "reply_count": 1}
        ],
        "has_more": False,
    }
    mock_client.conversations_replies.return_value = {
        "messages": [
            {"text": "Thread parent", "ts": "123", "user": "U123"},
            {"text": "Thread reply", "ts": "124", "user": "U124"},
        ]
    }

    messages = reader._get_channel_messages(mock_client, "C123", limit=10)
    assert len(messages) == 2


def test_load_data(mocker):
    """Test loading data from channels."""
    reader = SlackReader(slack_token="test-token")
    mock_client = mocker.Mock()
    mocker.patch("slack_sdk.WebClient", return_value=mock_client)

    # Mock successful channel check
    mock_client.conversations_info.return_value = {"ok": True}

    # Mock messages
    mock_client.conversations_history.return_value = {
        "messages": [
            {"text": "Hello", "ts": "123", "user": "U123"},
            {"text": "World", "ts": "124", "user": "U124"},
        ],
        "has_more": False,
    }

    docs = reader.load_data(channel_ids=["C123"], limit=10)
    assert len(docs) == 2
    assert docs[0].text == "Hello"

    # Test channel not found
    mock_client.conversations_info.side_effect = SlackApiError(
        "channel_not_found", {"error": "channel_not_found"}
    )
    with pytest.raises(ValueError, match="Channel .* not found"):
        reader.load_data(channel_ids=["C456"])

    # Test invalid channel ID type
    with pytest.raises(ValueError, match="Channel id .* must be a string"):
        reader.load_data(channel_ids=[123])


def test_thread_replies(mocker):
    """Test handling of thread replies."""
    reader = SlackReader(slack_token="xoxb-fake-token")
    mock_client = mocker.Mock()

    # Test parent message (thread_ts == ts)
    parent_msg = {
        "text": "Parent",
        "ts": "123.456",
        "thread_ts": "123.456",  # Same as ts = parent message
        "user": "U123",
        "reply_count": 2,
    }

    # Test thread replies
    mock_client.conversations_history.return_value = {
        "messages": [parent_msg],
        "has_more": False,
    }
    mock_client.conversations_replies.return_value = {
        "messages": [
            parent_msg,  # Parent included in replies
            {
                "text": "Reply 1",
                "ts": "123.457",
                "thread_ts": "123.456",
                "user": "U124",
            },
            {
                "text": "Reply 2",
                "ts": "123.458",
                "thread_ts": "123.456",
                "user": "U125",
            },
        ]
    }

    messages = reader._get_channel_messages(mock_client, "C123", limit=10)
    assert len(messages) == 3  # Parent + 2 replies

    # Test failed thread fetch
    mock_client.conversations_replies.side_effect = SlackApiError(
        "error", {"error": "thread_not_found"}
    )
    messages = reader._get_channel_messages(mock_client, "C123", limit=10)
    assert len(messages) == 1  # Only parent message


def test_message_metadata():
    """Test message metadata handling."""
    reader = SlackReader(slack_token="xoxb-fake-token")

    # Test edited message
    edited_msg = {
        "text": "Edited text",
        "ts": "123.456",
        "user": "U123",
        "edited": {"ts": "123.457"},
    }
    doc = reader._process_message(edited_msg, include_bots=True)
    assert doc.metadata["edited_at"] == "123.457"

    # Test message without channel
    no_channel_msg = {"text": "No channel", "ts": "123.456", "user": "U123"}
    doc = reader._process_message(no_channel_msg, include_bots=True)
    assert doc.metadata["channel_id"] == ""

    # Test message with channel
    channel_msg = {
        "text": "Has channel",
        "ts": "123.456",
        "user": "U123",
        "channel": "C123",
    }
    doc = reader._process_message(channel_msg, include_bots=True)
    assert doc.metadata["channel_id"] == "C123"


def test_pagination(mocker):
    """Test pagination handling."""
    reader = SlackReader(slack_token="xoxb-fake-token")
    mock_client = mocker.Mock()

    # Setup messages for two pages
    page1 = {
        "messages": [
            {"text": "Msg1", "ts": "123.456", "user": "U123"},
            {"text": "Msg2", "ts": "123.455", "user": "U124"},
        ],
        "has_more": True,  # More messages available
    }
    page2 = {
        "messages": [
            {"text": "Msg3", "ts": "123.454", "user": "U125"},
            {"text": "Msg4", "ts": "123.453", "user": "U126"},
        ],
        "has_more": False,
    }

    mock_client.conversations_history.side_effect = [page1, page2]

    messages = reader._get_channel_messages(mock_client, "C123", limit=None)
    assert len(messages) == 4
    assert messages[0]["text"] == "Msg1"
    assert messages[-1]["text"] == "Msg4"

    # Verify latest_ts was updated correctly
    calls = mock_client.conversations_history.call_args_list
    assert (
        calls[1][1]["latest"] == "123.456"
    )  # Second call should use ts from last message of first page


def test_limit_handling(mocker):
    """Test message limit handling."""
    reader = SlackReader(slack_token="xoxb-fake-token")
    mock_client = mocker.Mock()

    # Test limit=None with multiple pages
    page1 = {
        "messages": [
            {"text": f"Msg{i}", "ts": f"123.45{i}", "user": "U123"} for i in range(5)
        ],
        "has_more": True,
    }
    page2 = {
        "messages": [
            {"text": f"Msg{i}", "ts": f"123.44{i}", "user": "U123"} for i in range(5)
        ],
        "has_more": False,
    }
    mock_client.conversations_history.side_effect = [page1, page2]

    messages = reader._get_channel_messages(mock_client, "C123", limit=None)
    assert len(messages) == 10  # Should get all messages

    # Test exact limit
    mock_client.conversations_history.side_effect = None
    mock_client.conversations_history.return_value = page1
    messages = reader._get_channel_messages(mock_client, "C123", limit=3)
    assert len(messages) == 3  # Should only get first 3 messages

    # Test limit with thread replies
    thread_msg = {
        "messages": [
            {"text": "Parent", "ts": "123.456", "user": "U123", "reply_count": 2}
        ],
        "has_more": False,
    }
    mock_client.conversations_history.return_value = thread_msg
    mock_client.conversations_replies.return_value = {
        "messages": [
            {"text": "Parent", "ts": "123.456", "user": "U123"},
            {"text": "Reply1", "ts": "123.457", "user": "U124"},
            {"text": "Reply2", "ts": "123.458", "user": "U125"},
        ]
    }

    messages = reader._get_channel_messages(mock_client, "C123", limit=2)
    assert len(messages) == 2  # Should respect limit even with thread replies


def test_channel_validation():
    """Test channel ID validation."""
    reader = SlackReader(slack_token="xoxb-fake-token")

    # Test empty channel list
    with pytest.raises(ValueError):
        reader.load_data(channel_ids=[])

    # Test invalid channel format
    with pytest.raises(ValueError):
        reader.load_data(channel_ids=[123])  # Not a string

    # Test multiple channels
    with pytest.raises(ValueError):
        reader.load_data(channel_ids=["not-a-channel-id"])


def test_error_handling(mocker):
    """Test error handling."""
    reader = SlackReader(slack_token="xoxb-fake-token")

    # Test missing required field
    bad_msg = {"ts": "123.456", "user": "U123"}  # Missing 'text'
    with pytest.raises(KeyError):
        reader._process_message(bad_msg, include_bots=True)

    # Test malformed message
    malformed_msg = None
    with pytest.raises(TypeError):
        reader._process_message(malformed_msg, include_bots=True)

    # Test rate limiting
    mock_client = mocker.Mock()
    mock_client.conversations_history.side_effect = SlackApiError(
        "ratelimited", {"error": "ratelimited"}
    )
    messages = reader._get_channel_messages(mock_client, "C123", limit=10)
    assert len(messages) == 0


@pytest.fixture()
def mock_slack_import(mocker):
    """Mock slack_sdk import to fail."""
    return mocker.patch.dict("sys.modules", {"slack_sdk": None})


def test_import_error(mock_slack_import):
    """Test slack_sdk import error."""
    with pytest.raises(ImportError, match="Package `slack_sdk` not found"):
        SlackReader(slack_token="xoxb-fake-token")
