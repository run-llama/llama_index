from imap_tools import EmailAddress
from llama_index.core.schema import Document
from llama_index.readers.imap import ImapReader
import unittest.mock as mock


@mock.patch("llama_index.readers.imap.base.MailBox")
def test_connection(MockMailBox):
    ImapReader(host="test", username="test", password="test")

    MockMailBox.assert_called_once_with("test")
    MockMailBox.return_value.login.assert_called_once_with("test", "test")


def test_read():
    mock_msg1 = mock.MagicMock()
    mock_msg1.from_ = "john@doe.com"
    mock_msg1.to = ["jane@doe.com"]
    mock_msg1.to_values = EmailAddress("", "jane@doe.com")
    mock_msg1.from_values = EmailAddress("", "john@doe.com")
    mock_msg1.subject = "Test"
    mock_msg1.text = "Lorem ipsum dolor sit amet"
    mock_msg1.uid = 1

    mock_msg2 = mock.MagicMock()
    mock_msg2.from_ = "mark@doe.com"
    mock_msg2.to = ["jenna@doe.com", "joe@doe.com"]
    mock_msg2.to_values = EmailAddress("", "jenna@doe.com")
    mock_msg2.from_values = EmailAddress("", "mark@doe.com")
    mock_msg2.subject = "Test 2"
    mock_msg2.text = "Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet."
    mock_msg2.uid = 2

    with mock.patch("llama_index.readers.imap.base.MailBox") as MockMailBox:
        MockMailBox.return_value.fetch.return_value = [mock_msg1, mock_msg2]

        reader = ImapReader(host="test", username="test", password="test")

        messages = list(reader.lazy_load_data(metadata_names=["uid", "to", "text"]))
        assert len(messages) == 2

        assert isinstance(messages[0], Document)
        assert isinstance(messages[1], Document)

        # MESSAGE 1
        expected_text = "From: john@doe.com, To: jane@doe.com, Subject: Test, Message: Lorem ipsum dolor sit amet"
        assert messages[0].text == expected_text

        assert "uid" in messages[0].metadata
        assert messages[0].metadata["uid"] == 1

        assert "to" in messages[0].metadata
        assert messages[0].metadata["to"] == "jane@doe.com"

        assert "date" in messages[0].metadata

        # MESSAGE 2
        expected_text = "From: mark@doe.com, To: jenna@doe.com joe@doe.com, Subject: Test 2, Message: Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet."
        assert messages[1].text == expected_text

        assert "uid" in messages[1].metadata
        assert messages[1].metadata["uid"] == 2

        assert "to" in messages[1].metadata
        assert messages[1].metadata["to"] == "jenna@doe.com joe@doe.com"

        assert "date" in messages[1].metadata

        assert "email_text" in messages[0].metadata
