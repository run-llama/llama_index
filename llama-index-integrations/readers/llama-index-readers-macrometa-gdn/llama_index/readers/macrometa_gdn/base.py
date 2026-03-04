"""Macrometa GDN Reader."""

import json
from typing import List

import requests
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class MacrometaGDNReader(BaseReader):
    """
    Macrometa GDN Reader.

    Reads vectors from Macrometa GDN


    """

    def __init__(self, url: str, apikey: str):
        self.url = url
        self.apikey = apikey

    def load_data(self, collection_list: List[str]) -> List[Document]:
        """
        Loads data from the input directory.

        Args:
            api: Macrometa GDN API key
            collection_name: Name of the collection to read from

        """
        if collection_list is None:
            raise ValueError("Must specify collection name(s)")

        results = []
        for collection_name in collection_list:
            collection = self._load_collection(collection_name)
            results.append(
                Document(
                    text=collection, extra_info={"collection_name": collection_name}
                )
            )
        return results

    def _load_collection(self, collection_name: str) -> str:
        all_documents = []
        """Loads a collection from the database.

        Args:
            collection_name: Name of the collection to read from

        """
        url = self.url + "/_fabric/_system/_api/cursor"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": "apikey " + self.apikey,
        }

        data = {
            "batchSize": 1000,
            "ttl": 60,
            "query": "FOR doc IN " + collection_name + " RETURN doc",
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response_json = response.json()
        if response.status_code == 201:
            all_documents.extend(response_json.get("result", []))

            while response_json.get("hasMore"):
                cursor_id = response_json.get("id")

                next_url = self.url + "/_fabric/_system/_api/cursor/" + cursor_id

                response = requests.put(next_url, headers=headers)

                if response.status_code == 200:
                    response_json = response.json()
                    all_documents.extend(response_json.get("result", []))
                else:
                    print(f"Request failed with status code {response.status_code}")
                    break
        else:
            print(f"Initial request failed with status code {response.status_code}")

        return str(all_documents)


if __name__ == "__main__":
    reader = MacrometaGDNReader("https://api-anurag.eng.macrometa.io", "test")
    print(reader.load_data(collection_list=["test"]))
