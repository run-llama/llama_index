"""
Class for searching and importing data from OpenAlex.
"""

import logging
from typing import List

import requests
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


class OpenAlexReader(BaseReader):
    """
    This class is used to search and import data from OpenAlex.

    Parameters
    ----------
    email : str
        Email address to use for OpenAlex API

    Attributes
    ----------
    Works : pyalex.Works
        pyalex.Works object
    pyalex : pyalex
        pyalex object

    """

    def __init__(self, email) -> None:
        self.email = email

    def _search_openalex(self, query, fields):
        base_url = "https://api.openalex.org/works?search="
        fields_param = f"&select={fields}"
        email_param = f"&mailto={self.email}"
        full_url = base_url + query + fields_param + email_param
        try:
            response = requests.get(full_url, timeout=10)
            response.raise_for_status()  # Check if request is successful
            data = response.json()  # Parse JSON data
            if "error" in data:
                raise ValueError(f"API returned error: {data['error']}")
            return data
        except requests.exceptions.HTTPError as http_error:
            logger.error(f"HTTP error occurred: {http_error}")
        except requests.exceptions.RequestException as request_error:
            logger.error(f"Error occurred: {request_error}")
        except ValueError as value_error:
            logger.error(value_error)
        return None

    def _fulltext_search_openalex(self, query, fields):
        base_url = "https://api.openalex.org/works?filter=fulltext.search:"
        fields_param = f"&select={fields}"
        email_param = f"&mailto={self.email}"
        full_url = base_url + query + fields_param + email_param
        try:
            response = requests.get(full_url, timeout=10)
            response.raise_for_status()  # Check if request is successful
            data = response.json()  # Parse JSON data
            if "error" in data:
                raise ValueError(f"API returned error: {data['error']}")
            return data
        except requests.exceptions.HTTPError as http_error:
            logger.error(f"HTTP error occurred: {http_error}")
        except requests.exceptions.RequestException as request_error:
            logger.error(f"Error occurred: {request_error}")
        except ValueError as value_error:
            logger.error(value_error)
        return None

    def _invert_abstract(self, inv_index):
        if inv_index is not None:
            l_inv = [(w, p) for w, pos in inv_index.items() for p in pos]
            return " ".join(x[0] for x in sorted(l_inv, key=lambda x: x[1]))
        return None

    def load_data(self, query: str, full_text=False, fields=None) -> List[Document]:
        if fields is None:
            fields = "title,abstract_inverted_index,publication_year,keywords,authorships,primary_location"

        if full_text:
            works = self._fulltext_search_openalex(query, fields)
        else:
            works = self._search_openalex(query, fields)

        documents = []

        for work in works["results"]:
            if work["abstract_inverted_index"] is not None:
                abstract = self._invert_abstract(work["abstract_inverted_index"])
            else:
                abstract = None
            title = work.get("title", None)
            text = None
            # concat title and abstract
            if abstract and title:
                text = title + " " + abstract
            elif not abstract:
                text = title
            try:
                primary_location = work["primary_location"]["source"]["display_name"]
            except (KeyError, TypeError):
                primary_location = None

            metadata = {
                "title": work.get("title", None),
                "keywords": work.get("keywords", None),
                "primary_location": primary_location,
                "publication_year": work.get("publication_year", None),
                "authorships": [
                    item["author"]["display_name"] for item in work["authorships"]
                ],
            }

            documents.append(Document(text=text, extra_info=metadata))

        return documents
