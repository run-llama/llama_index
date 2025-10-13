"""monday.com reader."""

from typing import Dict, List

import requests
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class MondayReader(BaseReader):
    """
    monday.com reader. Reads board's data by a GraphQL query.

    Args:
        api_key (str): monday.com API key.

    """

    def __init__(self, api_key: str) -> None:
        """Initialize monday.com reader."""
        self.api_key = api_key
        self.api_url = "https://api.monday.com/v2"

    def _parse_item_values(self, cv) -> Dict[str, str]:
        data = {}
        data["title"] = cv["title"]
        data["value"] = cv["text"]

        return data

    def _parse_data(self, item) -> Dict[str, str]:
        data = {}
        data["id"] = item["id"]
        data["name"] = item["name"]
        data["values"] = list(map(self._parse_item_values, list(item["column_values"])))

        return data

    def _perform_request(self, board_id) -> Dict[str, str]:
        headers = {"Authorization": self.api_key}
        query = """
            query{
                boards(ids: [%d]){
                    name,
                    items{
                        id,
                        name,
                        column_values{
                            title,
                            text
                        }
                    }
                }
            } """ % (board_id)
        data = {"query": query}

        response = requests.post(url=self.api_url, json=data, headers=headers)
        return response.json()

    def load_data(self, board_id: int) -> List[Document]:
        """
        Load board data by board_id.

        Args:
            board_id (int): monday.com board id.


        Returns:
            List[Document]: List of items as documents.
            [{id, name, values: [{title, value}]}]

        """
        json_response = self._perform_request(board_id)
        board_data = json_response["data"]["boards"][0]

        board_data["name"]
        items_array = list(board_data["items"])
        parsed_items = list(map(self._parse_data, list(items_array)))
        result = []
        for item in parsed_items:
            text = f"name: {item['name']}"
            for item_value in item["values"]:
                if item_value["value"]:
                    text += f", {item_value['title']}: {item_value['value']}"
            result.append(
                Document(
                    text=text, extra_info={"board_id": board_id, "item_id": item["id"]}
                )
            )

        return result


if __name__ == "__main__":
    reader = MondayReader("api_key")
    print(reader.load_data(12345))
