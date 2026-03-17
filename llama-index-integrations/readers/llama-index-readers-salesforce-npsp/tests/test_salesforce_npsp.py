"""
Tests for SalesforceNPSPReader.
All tests mock simple_salesforce.Salesforce — no real credentials needed.
"""

from unittest.mock import MagicMock
import pytest

from llama_index.readers.salesforce_npsp import SalesforceNPSPReader
from llama_index.core.schema import Document


MOCK_CONTACTS = [
    {
        "Id": "003XXXXXXXXXXXXXXX",
        "FirstName": "Jane",
        "LastName": "Smith",
        "Email": "jane@example.org",
        "Title": None,
        "npo02__TotalOppAmount__c": 50000.0,
        "npo02__NumberOfClosedOpps__c": 5.0,
        "npo02__LastCloseDate__c": "2024-02-15",
        "npo02__FirstCloseDate__c": "2018-06-01",
        "npo02__AverageAmount__c": 10000.0,
        "npo02__LargestAmount__c": 25000.0,
        "npo02__LastMembershipDate__c": None,
        "npsp__Primary_Affiliation__r": {"Name": "City Hospital"},
        "npsp__Soft_Credit_Total__c": 5000.0,
        "npsp__Planned_Giving_Count__c": 1.0,
        "LastActivityDate": "2023-08-01",
        "CreatedDate": "2018-05-15T10:00:00.000+0000",
    }
]

MOCK_OPPORTUNITIES = [
    {
        "Id": "006XXXXXXXXXXXXXXX",
        "Name": "Jane Smith Major Gift",
        "Amount": 25000.0,
        "CloseDate": "2024-02-15",
        "StageName": "Closed Won",
        "RecordType": {"Name": "Major Gift"},
        "npsp__Acknowledgment_Status__c": "Acknowledged",
        "npsp__Gift_Strategy__c": "Major Gift",
        "Primary_Contact__c": "003XXXXXXXXXXXXXXX",
    }
]


@pytest.fixture
def mock_sf():
    sf = MagicMock()
    sf.query_all.side_effect = [
        {"records": MOCK_CONTACTS},
        {"records": MOCK_OPPORTUNITIES},
    ]
    return sf


@pytest.fixture
def reader():
    return SalesforceNPSPReader(
        username="test@example.org",
        password="password",
        security_token="TOKEN",
        domain="test",
    )


def test_reader_init_from_args():
    r = SalesforceNPSPReader(username="u", password="p", security_token="t")
    assert r.username == "u"
    assert r.include_opportunities is True


def test_reader_init_from_env(monkeypatch):
    monkeypatch.setenv("SF_USERNAME", "env_user")
    monkeypatch.setenv("SF_PASSWORD", "env_pass")
    monkeypatch.setenv("SF_TOKEN", "env_token")
    r = SalesforceNPSPReader()
    assert r.username == "env_user"


def test_missing_credentials_raises(monkeypatch):
    monkeypatch.delenv("SF_USERNAME", raising=False)
    monkeypatch.delenv("SF_PASSWORD", raising=False)
    monkeypatch.delenv("SF_TOKEN", raising=False)
    with pytest.raises(ValueError, match="credentials"):
        SalesforceNPSPReader()


def test_load_data_returns_documents(reader, mock_sf):
    reader._sf = mock_sf
    docs = reader.load_data()
    assert isinstance(docs, list)
    assert len(docs) == 1
    assert isinstance(docs[0], Document)


def test_document_text_contains_donor_name(reader, mock_sf):
    reader._sf = mock_sf
    docs = reader.load_data()
    assert "Jane Smith" in docs[0].text


def test_document_text_contains_gift_history(reader, mock_sf):
    reader._sf = mock_sf
    docs = reader.load_data()
    assert "$25,000" in docs[0].text


def test_document_metadata_keys(reader, mock_sf):
    reader._sf = mock_sf
    docs = reader.load_data()
    expected = {
        "donor_id",
        "donor_name",
        "email",
        "affiliation",
        "total_gift_amount",
        "gift_count",
        "average_gift_amount",
        "largest_gift_amount",
        "first_gift_date",
        "last_gift_date",
        "last_activity_date",
        "soft_credit_total",
        "planned_giving_count",
        "source",
    }
    assert expected == set(docs[0].metadata.keys())


def test_document_metadata_values(reader, mock_sf):
    reader._sf = mock_sf
    docs = reader.load_data()
    assert docs[0].metadata["total_gift_amount"] == 50000.0
    assert docs[0].metadata["gift_count"] == 5
    assert docs[0].metadata["source"] == "salesforce_npsp"


def test_affinity_score_injected(mock_sf):
    reader = SalesforceNPSPReader(
        username="u",
        password="p",
        security_token="t",
        affinity_score_fn=lambda meta: 87.5,
    )
    mock_sf.query_all.side_effect = [
        {"records": MOCK_CONTACTS},
        {"records": MOCK_OPPORTUNITIES},
    ]
    reader._sf = mock_sf
    docs = reader.load_data()
    assert docs[0].metadata["affinity_score"] == 87.5


def test_affinity_score_exception_handled(mock_sf):
    def bad_scorer(meta):
        raise RuntimeError("model not fitted")

    reader = SalesforceNPSPReader(
        username="u",
        password="p",
        security_token="t",
        affinity_score_fn=bad_scorer,
    )
    mock_sf.query_all.side_effect = [
        {"records": MOCK_CONTACTS},
        {"records": MOCK_OPPORTUNITIES},
    ]
    reader._sf = mock_sf
    docs = reader.load_data()
    assert docs[0].metadata["affinity_score"] is None


def test_empty_contacts_returns_empty_list(reader):
    mock_sf = MagicMock()
    mock_sf.query_all.return_value = {"records": []}
    reader._sf = mock_sf
    docs = reader.load_data()
    assert docs == []


def test_no_opportunities_mode():
    reader = SalesforceNPSPReader(
        username="u",
        password="p",
        security_token="t",
        include_opportunities=False,
    )
    mock_sf = MagicMock()
    mock_sf.query_all.return_value = {"records": MOCK_CONTACTS}
    reader._sf = mock_sf
    reader.load_data()
    assert mock_sf.query_all.call_count == 1


def test_contact_ids_filter(reader, mock_sf):
    reader._sf = mock_sf
    reader.load_data(contact_ids=["003XXXXXXXXXXXXXXX"])
    soql_used = mock_sf.query_all.call_args_list[0][0][0]
    assert "003XXXXXXXXXXXXXXX" in soql_used


def test_missing_npsp_fields_handled_gracefully(reader):
    sparse = {
        "Id": "003YYY",
        "FirstName": None,
        "LastName": None,
        "Email": None,
        "Title": None,
        "npo02__TotalOppAmount__c": None,
        "npo02__NumberOfClosedOpps__c": None,
        "npo02__LastCloseDate__c": None,
        "npo02__FirstCloseDate__c": None,
        "npo02__AverageAmount__c": None,
        "npo02__LargestAmount__c": None,
        "npo02__LastMembershipDate__c": None,
        "npsp__Primary_Affiliation__r": None,
        "npsp__Soft_Credit_Total__c": None,
        "npsp__Planned_Giving_Count__c": None,
        "LastActivityDate": None,
        "CreatedDate": None,
    }
    mock_sf = MagicMock()
    mock_sf.query_all.return_value = {"records": [sparse]}
    reader._sf = mock_sf
    docs = reader.load_data()
    assert len(docs) == 1
    assert docs[0].metadata["total_gift_amount"] == 0.0


def test_soql_filter_passed_through(reader, mock_sf):
    reader._sf = mock_sf
    reader.load_data(soql_filter="npo02__TotalOppAmount__c > 10000", limit=50)
    soql_used = mock_sf.query_all.call_args_list[0][0][0]
    assert "npo02__TotalOppAmount__c > 10000" in soql_used
    assert "LIMIT 50" in soql_used


def test_contact_ids_ignores_soql_filter_and_limit(reader, mock_sf):
    reader._sf = mock_sf
    reader.load_data(
        contact_ids=["003XXXXXXXXXXXXXXX"],
        soql_filter="npo02__TotalOppAmount__c > 99999",
        limit=1,
    )
    soql_used = mock_sf.query_all.call_args_list[0][0][0]
    # The explicit ID must appear
    assert "003XXXXXXXXXXXXXXX" in soql_used
    # The soql_filter text must NOT appear
    assert "npo02__TotalOppAmount__c > 99999" not in soql_used
    # LIMIT must NOT appear (no silent ID truncation)
    assert "LIMIT" not in soql_used
