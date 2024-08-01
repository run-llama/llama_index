"""Notion tool spec."""

from typing import Any, Dict, List, Optional, Type

import requests

from llama_index.legacy.bridge.pydantic import BaseModel
from llama_index.legacy.readers.notion import NotionPageReader
from llama_index.legacy.tools.tool_spec.base import SPEC_FUNCTION_TYPE, BaseToolSpec

SEARCH_URL = "https://api.notion.com/v1/search"


class NotionLoadDataSchema(BaseModel):
    """Notion load data schema."""

    page_ids: Optional[List[str]] = None
    database_id: Optional[str] = None


class NotionSearchDataSchema(BaseModel):
    """Notion search data schema."""

    query: str
    direction: Optional[str] = None
    timestamp: Optional[str] = None
    value: Optional[str] = None
    property: Optional[str] = None
    page_size: int = 100


class NotionToolSpec(BaseToolSpec):
    """Notion tool spec.

    Currently a simple wrapper around the data loader.
    TODO: add more methods to the Notion spec.

    """

    spec_functions = ["load_data", "search_data"]

    def __init__(self, integration_token: Optional[str] = None) -> None:
        """Initialize with parameters."""
        self.reader = NotionPageReader(integration_token=integration_token)

    def get_fn_schema_from_fn_name(
        self, fn_name: str, spec_functions: Optional[List[SPEC_FUNCTION_TYPE]] = None
    ) -> Optional[Type[BaseModel]]:
        """Return map from function name."""
        if fn_name == "load_data":
            return NotionLoadDataSchema
        elif fn_name == "search_data":
            return NotionSearchDataSchema
        else:
            raise ValueError(f"Invalid function name: {fn_name}")

    def load_data(
        self, page_ids: Optional[List[str]] = None, database_id: Optional[str] = None
    ) -> str:
        """Loads content from a set of page ids or a database id.

        Don't use this endpoint if you don't know the page ids or database id.

        """
        page_ids = page_ids or []
        docs = self.reader.load_data(page_ids=page_ids, database_id=database_id)
        return "\n".join([doc.get_content() for doc in docs])

    def search_data(
        self,
        query: str,
        direction: Optional[str] = None,
        timestamp: Optional[str] = None,
        value: Optional[str] = None,
        property: Optional[str] = None,
        page_size: int = 100,
    ) -> str:
        """Search a list of relevant pages.

        Contains metadata for each page (but not the page content).

        """
        payload: Dict[str, Any] = {
            "query": query,
            "page_size": page_size,
        }
        if direction is not None or timestamp is not None:
            payload["sort"] = {}
            if direction is not None:
                payload["sort"]["direction"] = direction
            if timestamp is not None:
                payload["sort"]["timestamp"] = timestamp

        if value is not None or property is not None:
            payload["filter"] = {}
            if value is not None:
                payload["filter"]["value"] = value
            if property is not None:
                payload["filter"]["property"] = property

        response = requests.post(SEARCH_URL, json=payload, headers=self.reader.headers)
        response_json = response.json()
        return response_json["results"]
