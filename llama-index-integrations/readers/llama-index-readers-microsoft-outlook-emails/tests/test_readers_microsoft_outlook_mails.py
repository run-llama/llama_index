from unittest.mock import patch

import pytest
from llama_index.core.readers.base import BaseReader
from llama_index.readers.microsoft_outlook_emails import OutlookEmailReader


def test_class():
    names_of_base_classes = [b.__name__ for b in OutlookEmailReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_serialize():
    reader = OutlookEmailReader(
        client_id="test_client_id",
        client_secret="test_client_secret",
        tenant_id="test_tenant_id",
        user_email="test_user@domain.com",
        num_mails=5,
    )

    # Get the JSON schema
    schema = reader.model_json_schema()
    assert schema is not None
    assert "properties" in schema
    assert "client_id" in schema["properties"]
    assert "client_secret" in schema["properties"]
    assert "tenant_id" in schema["properties"]
    assert "user_email" in schema["properties"]

    # Test serialization/deserialization
    data = reader.model_dump()
    data.pop("is_remote")
    data.pop("class_name")
    new_reader = OutlookEmailReader.model_validate(data)
    assert new_reader.client_id == reader.client_id
    assert new_reader.client_secret == reader.client_secret
    assert new_reader.tenant_id == reader.tenant_id
    assert new_reader.user_email == reader.user_email


@pytest.fixture()
def outlook_reader():
    return OutlookEmailReader(
        client_id="dummy_client_id",
        client_secret="dummy_client_secret",
        tenant_id="dummy_tenant_id",
        user_email="dummy_user@domain.com",
        num_mails=2,
    )


def mock_fetch_emails(*args, **kwargs):
    return [
        {
            "subject": "Test Email 1",
            "body": {"content": "This is the body of email 1."},
        },
        {
            "subject": "Test Email 2",
            "body": {"content": "This is the body of email 2."},
        },
    ]


def test_load_data(outlook_reader):
    # Only mock the response from _fetch_emails, not the entire method
    with patch.object(
        OutlookEmailReader, "_fetch_emails", return_value=mock_fetch_emails()
    ):
        email_texts = outlook_reader.load_data()

        # Verify the results
        assert len(email_texts) == 2
        assert email_texts[0] == "Subject: Test Email 1\n\nThis is the body of email 1."
        assert email_texts[1] == "Subject: Test Email 2\n\nThis is the body of email 2."
