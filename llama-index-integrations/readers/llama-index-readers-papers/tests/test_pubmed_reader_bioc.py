"""Regression tests for PubmedReader.load_data_bioc title extraction.

The BioC payloads are captured from the PMC BioC API
(https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/BioC-PMC/) and served by a
stubbed ``requests.get``, so these tests need no network access.
"""

from unittest.mock import Mock

from llama_index.readers.papers import PubmedReader

ESEARCH_XML = b"""<?xml version="1.0" ?>
<eSearchResult>
  <Count>1</Count>
  <IdList><Id>9000001</Id></IdList>
</eSearchResult>
"""

BIOC_PAYLOAD_WITH_TITLE = {
    "source": "PMC",
    "date": "20240102",
    "documents": [
        {
            "id": "PMC9000001",
            "passages": [
                {
                    "infons": {"section_type": "TITLE", "type": "front"},
                    "text": "Alzheimer disease: a review of the literature",
                },
                {
                    "infons": {"section_type": "ABSTRACT", "type": "abstract"},
                    "text": "Amyloid plaques are the hallmark of the disease.",
                },
            ],
        }
    ],
}

BIOC_PAYLOAD_WITHOUT_TITLE = {
    "source": "PMC",
    "date": "20240102",
    "documents": [
        {
            "id": "PMC9000001",
            "passages": [
                {
                    "infons": {"section_type": "ABSTRACT", "type": "abstract"},
                    "text": "Amyloid plaques are the hallmark of the disease.",
                }
            ],
        }
    ],
}


def _fake_response(content: bytes = b"", json_data=None) -> Mock:
    response = Mock()
    response.content = content
    response.status_code = 200
    response.json.return_value = json_data
    return response


def _stub_requests(monkeypatch, bioc_payload) -> None:
    def fake_get(url, **kwargs):
        if "esearch.fcgi" in url:
            return _fake_response(content=ESEARCH_XML)
        return _fake_response(json_data=bioc_payload)

    monkeypatch.setattr("requests.get", fake_get)


def test_load_data_bioc_returns_the_paper(monkeypatch):
    _stub_requests(monkeypatch, BIOC_PAYLOAD_WITH_TITLE)

    documents = PubmedReader().load_data_bioc("alzheimers", max_results=1)

    assert len(documents) == 1
    assert (
        documents[0].metadata["Title of this paper"]
        == "Alzheimer disease: a review of the literature"
    )
    assert documents[0].metadata["URL"].endswith("PMC9000001/")
    assert documents[0].metadata["Date published"] == "01/02/2024"
    assert "Amyloid plaques" in documents[0].text


def test_load_data_bioc_falls_back_to_default_title(monkeypatch):
    _stub_requests(monkeypatch, BIOC_PAYLOAD_WITHOUT_TITLE)

    documents = PubmedReader().load_data_bioc("alzheimers", max_results=1)

    assert len(documents) == 1
    assert documents[0].metadata["Title of this paper"] == "Pubmed Paper"
