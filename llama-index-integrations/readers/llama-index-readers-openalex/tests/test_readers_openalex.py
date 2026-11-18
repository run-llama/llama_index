from llama_index.core.readers.base import BaseReader
from llama_index.readers.openalex import OpenAlexReader


def test_class():
    names_of_base_classes = [b.__name__ for b in OpenAlexReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_search_uses_request_params(mocker):
    response = mocker.Mock()
    response.json.return_value = {"results": []}
    request = mocker.patch("llama_index.readers.openalex.base.requests.get")
    request.return_value = response

    reader = OpenAlexReader(email="research+openalex@example.com")
    reader._search_openalex("C++ & retrieval", "title,publication_year")

    request.assert_called_once_with(
        "https://api.openalex.org/works",
        params={
            "search": "C++ & retrieval",
            "select": "title,publication_year",
            "mailto": "research+openalex@example.com",
        },
        timeout=10,
    )


def test_fulltext_search_uses_request_params(mocker):
    response = mocker.Mock()
    response.json.return_value = {"results": []}
    request = mocker.patch("llama_index.readers.openalex.base.requests.get")
    request.return_value = response

    reader = OpenAlexReader(email="research+openalex@example.com")
    reader._fulltext_search_openalex("agent safety & control", "title")

    request.assert_called_once_with(
        "https://api.openalex.org/works",
        params={
            "filter": "fulltext.search:agent safety & control",
            "select": "title",
            "mailto": "research+openalex@example.com",
        },
        timeout=10,
    )
