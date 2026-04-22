"""Patentsview reader that reads patent abstract."""

from typing import List

import os
import logging
import requests
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from typing import Optional

logger = logging.getLogger(__name__)
BASE_URL = "https://search.patentsview.org/api/v1/patent"


class PatentsviewReader(BaseReader):
    """
    Patentsview reader.

    Read patent abstract.

    """

    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        """Initialize with request body."""
        self.json = {
            "q": {"patent_id": None},
            "f": ["patent_id", "patent_abstract"],
            "o": {"size": 1000},  # API's max return
        }

        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("PATENTSVIEW_API_KEY", None)
            if self.api_key is None:
                raise ValueError("The API key [PATENTSVIEW_API_KEY] is required.")

        self.headers = {"X-Api-Key": self.api_key}

    def load_data(self, patent_number: List[str]) -> List[Document]:
        """
        Load patent abstract given list of patent numbers.

        Args:
            patent_number: List[str]: List of patent numbers, e.g., 8848839.

        Returns:
            List[Document]: A list of Document objects, each including the abstract for a patent.

        """
        if not patent_number:
            raise ValueError("Please input patent number")

        if len(patent_number) > 1000:
            raise ValueError(
                f"List patent number size is too large: {len(patent_number)} elements. Maximum allowed is 1000."
            )

        self.json["q"]["patent_id"] = patent_number

        response = requests.post(BASE_URL, json=self.json, headers=self.headers)
        if response.status_code == 429:
            wait = int(response.headers.get("Retry-After", 60))
            logging.info(f"Throttled. Retrying in {wait}s...")
            time.sleep(wait)
            response = requests.post(BASE_URL, json=self.json, headers=self.headers)

        if response.status_code == 200:
            data = response.json()
            patents = data.get("patents", [])

            results = []
            for patent in patents:
                metadata = {"patent_id": patent["patent_id"]}
                results.append(
                    Document(text=patent["patent_abstract"], metadata=metadata)
                )

        else:
            raise Exception(f"Request failed with status code: {response.status_code}")

        return results
