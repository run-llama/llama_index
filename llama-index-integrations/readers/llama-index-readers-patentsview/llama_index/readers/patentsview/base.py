"""Patentsview reader that reads patent abstract."""

from typing import List

import requests
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

BASE_URL = "https://api.patentsview.org/patents/query"


class PatentsviewReader(BaseReader):
    """
    Patentsview reader.

    Read patent abstract.

    """

    def __init__(self) -> None:
        """Initialize with request body."""
        self.json = {"q": {"patent_id": None}, "f": ["patent_abstract"]}

    def load_data(self, patent_number: List[str]) -> List[Document]:
        """
        Load patent abstract given list of patent numbers.

        Args:
            patent_number: List[str]: List of patent numbers, e.g., 8848839.

        Returens:
            List[Document]: A list of Document objects, each including the abstract for a patent.

        """
        if not patent_number:
            raise ValueError("Please input patent number")

        self.json["q"]["patent_id"] = patent_number

        response = requests.post(BASE_URL, json=self.json)

        if response.status_code == 200:
            data = response.json()
            patents = data.get("patents", [])

            results = []
            for patent in patents:
                results.append(Document(text=patent["patent_abstract"]))

        else:
            raise Exception(f"Request failed with status code: {response.status_code}")

        return results
