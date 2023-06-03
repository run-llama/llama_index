"""Notion tool spec."""

from llama_index.tools.tool_spec.base import BaseToolSpec
from llama_index.readers.notion import NotionPageReader
from typing import Optional, List, Type
import requests
from pydantic import BaseModel

SEARCH_URL = "https://api.notion.com/v1/search"


class NotionSort(BaseModel):
    """Notion sort."""

    direction: str
    timestamp: str


class NotionFilter(BaseModel):
    """Notion filter."""

    value: str
    property: str


class NotionSearchDataSchema(BaseModel):
    """Notion search data schema."""

    query: str
    sort: Optional[NotionSort] = None
    filter: Optional[NotionFilter] = None
    page_size: int = 100


class NotionToolSpec(BaseToolSpec):
    """Notion tool spec.

    Currently a simple wrapper around the data loader.
    TODO: add more methods to the Notion spec.

    """

    spec_functions = ["load_data", "search_data"]
    # spec_map = {
    #     "load_data": {"fn": "load_data", "fn_schema": "load_data_schema"},
    #     "search_data": {"fn": "search_data", "fn_schema": "search_data_schema"},
    # }

    def __init__(self, integration_token: Optional[str] = None) -> None:
        """Initialize with parameters."""
        self.reader = NotionPageReader(integration_token=integration_token)

    def get_fn_schema_from_fn_name(self, fn_name: str) -> Type[BaseModel]:
        """Return map from function name."""
        if fn_name == "load_data":
            pass
        elif fn_name == "search_data":
            return NotionSearchDataSchema
        else:
            raise ValueError(f"Invalid function name: {fn_name}")

    def load_data(
        self, page_ids: Optional[List[str]] = None, database_id: Optional[str] = None
    ) -> str:
        """Loads content from a set of page ids or a database id.

        Args:
            page_ids (List[str]): List of page ids to load.
            database_id (str): Database id to load.

        Returns:
            str: Loaded data.

        """
        page_ids = page_ids or []
        docs = self.reader.load_data(page_ids=page_ids, database_id=database_id)
        return "\n".join([doc.get_text() for doc in docs])

    def search_data(
        self,
        query: str,
        sort: Optional[NotionSort] = None,
        filter: Optional[NotionFilter] = None,
        page_size: int = 100,
    ) -> str:
        """Returns a list of relevant pages.

        Contains metadata for each page (but not the page content).

        """
        payload = {
            "query": query,
            "page_size": page_size,
        }
        if sort is not None:
            payload["sort"] = sort.dict()
        if filter is not None:
            payload["filter"] = filter.dict()

        response = requests.post(SEARCH_URL, json=payload, headers=self.reader.headers)
        response_json = response.json()
        response_results = response_json["results"]
        return response_results

    def search_data_schema(self) -> BaseModel:
        """Return search data schema."""
        return NotionSearchDataSchema
