"""Regression tests for the RCSB fallback of the PDB reader.

The responses served by the stubbed ``requests.get`` follow the shapes of
https://data.rcsb.org/rest/v1/core/entry/{id} and
https://data.rcsb.org/rest/v1/core/pubmed/{id}, so no network access is
needed.
"""

from unittest.mock import Mock

import pytest

from llama_index.readers.pdb import PdbAbstractReader
from llama_index.readers.pdb import utils as pdb_utils

RCSB_ENTRY = {
    "rcsb_id": "1ABC",
    "citation": [
        {"id": "primary", "title": "Structure of a test protein"},
        {"id": "secondary", "title": "An earlier related paper"},
    ],
}

RCSB_PUBMED = {
    "rcsb_pubmed_doi": "10.1000/test.doi",
    "rcsb_pubmed_abstract_text": "The abstract of the primary citation.",
}


def _response(status_code: int = 200, json_data=None) -> Mock:
    response = Mock()
    response.status_code = status_code
    response.json.return_value = json_data
    return response


def test_parse_rcsb_publication_dict_selects_the_primary_citation():
    title, publications = pdb_utils.parse_rcsb_publication_dict(
        RCSB_ENTRY, RCSB_PUBMED
    )

    assert title == "Structure of a test protein"
    assert publications[title] == {
        "doi": "10.1000/test.doi",
        "abstract": {"abstract": "The abstract of the primary citation."},
        "primary": True,
    }


def test_parse_rcsb_publication_dict_without_primary_citation():
    entry_without_primary = {"citation": [{"id": "secondary", "title": "Other"}]}

    with pytest.raises(ValueError):
        pdb_utils.parse_rcsb_publication_dict(entry_without_primary, RCSB_PUBMED)


def test_load_data_uses_the_rcsb_fallback_when_ebi_fails(monkeypatch):
    def fake_get(url, *args, **kwargs):
        if url.startswith("https://www.ebi.ac.uk/pdbe/api/pdb/entry/publications/"):
            return _response(status_code=404)
        if "/core/entry/" in url:
            return _response(json_data=RCSB_ENTRY)
        if "/core/pubmed/" in url:
            return _response(json_data=RCSB_PUBMED)
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr("requests.get", fake_get)

    documents = PdbAbstractReader().load_data(["1abc"])

    assert len(documents) == 1
    assert documents[0].metadata["pdb_id"] == "1abc"
    assert documents[0].metadata["primary_citation"]["doi"] == "10.1000/test.doi"
    assert "The abstract of the primary citation." in documents[0].text
