"""dad_jokes reader."""

from typing import List

import requests
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class DadJokesReader(BaseReader):
    """
    Dad jokes reader.

    Reads a random dad joke.

    """

    def _get_random_dad_joke(self):
        response = requests.get(
            "https://icanhazdadjoke.com/", headers={"Accept": "application/json"}
        )
        response.raise_for_status()
        json_data = response.json()
        return json_data["joke"]

    def load_data(self) -> List[Document]:
        """
        Return a random dad joke.

        Args:
            None.

        """
        return [Document(text=self._get_random_dad_joke())]
