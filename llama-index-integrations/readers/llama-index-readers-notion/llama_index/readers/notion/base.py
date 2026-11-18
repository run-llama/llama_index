"""Notion reader."""

import os
import time
from typing import Any, Dict, List, Optional

import requests  # type: ignore
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

INTEGRATION_TOKEN_NAME = "NOTION_INTEGRATION_TOKEN"
BLOCK_CHILD_URL_TMPL = "https://api.notion.com/v1/blocks/{block_id}/children"
DATABASE_URL_TMPL = "https://api.notion.com/v1/databases/{database_id}/query"
SEARCH_URL = "https://api.notion.com/v1/search"


# TODO: Notion DB reader coming soon!
class NotionPageReader(BasePydanticReader):
    """
    Notion Page reader.

    Reads a set of Notion pages.

    Args:
        integration_token (str): Notion integration token.

    """

    is_remote: bool = True
    token: str
    headers: Dict[str, str]

    def __init__(self, integration_token: Optional[str] = None) -> None:
        """Initialize with parameters."""
        if integration_token is None:
            integration_token = os.getenv(INTEGRATION_TOKEN_NAME)
            if integration_token is None:
                raise ValueError(
                    "Must specify `integration_token` or set environment "
                    "variable `NOTION_INTEGRATION_TOKEN`."
                )

        token = integration_token
        headers = {
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
        }

        super().__init__(token=token, headers=headers)

    @classmethod
    def class_name(cls) -> str:
        """Get the name identifier of the class."""
        return "NotionPageReader"

    @staticmethod
    def _get_block_text(block_obj: Dict[str, Any]) -> List[str]:
        """
        Return the text pieces of a block's type-specific object.

        Uses ``plain_text`` so that every rich text type is kept (``text``,
        ``mention`` and ``equation``), and reads ``table_row`` blocks, whose
        text lives in ``cells`` rather than ``rich_text``.
        """
        if "rich_text" in block_obj:
            return [rt.get("plain_text", "") for rt in block_obj["rich_text"]]
        if "cells" in block_obj:
            cells = [
                "".join(rt.get("plain_text", "") for rt in cell)
                for cell in block_obj["cells"]
            ]
            return [" | ".join(cells)]
        return []

    def _read_block(self, block_id: str, num_tabs: int = 0) -> str:
        """Read a block and its children, following pagination."""
        result_lines_arr = []
        block_url = BLOCK_CHILD_URL_TMPL.format(block_id=block_id)
        start_cursor: Optional[str] = None
        while True:
            # The children endpoint is paginated (max 100 per page). The next
            # page is requested from the same block with `start_cursor`.
            params: Dict[str, Any] = {"page_size": 100}
            if start_cursor is not None:
                params["start_cursor"] = start_cursor

            res = self._request_with_retry(
                "GET", block_url, headers=self.headers, params=params
            )
            data = res.json()

            for result in data["results"]:
                result_type = result["type"]
                result_obj = result[result_type]

                prefix = "\t" * num_tabs
                cur_result_text_arr = [
                    prefix + text for text in self._get_block_text(result_obj)
                ]

                result_block_id = result["id"]
                has_children = result["has_children"]
                if has_children:
                    children_text = self._read_block(
                        result_block_id, num_tabs=num_tabs + 1
                    )
                    cur_result_text_arr.append(children_text)

                cur_result_text = "\n".join(cur_result_text_arr)
                result_lines_arr.append(cur_result_text)

            if not data.get("has_more") or data.get("next_cursor") is None:
                break
            start_cursor = data["next_cursor"]

        return "\n".join(result_lines_arr)

    def _request_with_retry(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> requests.Response:
        """Make a request with retry and rate limit handling."""
        max_retries = 5
        backoff_factor = 1

        for attempt in range(max_retries):
            try:
                response = requests.request(
                    method, url, headers=headers, json=json, params=params
                )
                response.raise_for_status()
                return response
            except requests.exceptions.HTTPError:
                if response.status_code == 429:
                    # Rate limit exceeded
                    retry_after = int(response.headers.get("Retry-After", 1))
                    time.sleep(backoff_factor * (2**attempt) + retry_after)
                else:
                    raise requests.exceptions.HTTPError(
                        f"Request failed: {response.text}"
                    )
            except requests.exceptions.RequestException as err:
                raise requests.exceptions.RequestException(f"Request failed: {err}")
        raise Exception("Maximum retries exceeded")

    def read_page(self, page_id: str) -> str:
        """Read a page."""
        return self._read_block(page_id)

    def query_database(
        self, database_id: str, query_dict: Dict[str, Any] = {"page_size": 100}
    ) -> List[str]:
        """Get all the pages from a Notion database."""
        pages = []

        res = self._request_with_retry(
            "POST",
            DATABASE_URL_TMPL.format(database_id=database_id),
            headers=self.headers,
            json=query_dict,
        )
        res.raise_for_status()
        data = res.json()

        pages.extend(data.get("results"))

        while data.get("has_more"):
            query_dict["start_cursor"] = data.get("next_cursor")
            res = self._request_with_retry(
                "POST",
                DATABASE_URL_TMPL.format(database_id=database_id),
                headers=self.headers,
                json=query_dict,
            )
            res.raise_for_status()
            data = res.json()
            pages.extend(data.get("results"))

        return [page["id"] for page in pages]

    def search(self, query: str) -> List[str]:
        """Search Notion page given a text query."""
        done = False
        next_cursor: Optional[str] = None
        page_ids = []
        while not done:
            query_dict = {
                "query": query,
            }
            if next_cursor is not None:
                query_dict["start_cursor"] = next_cursor
            res = self._request_with_retry(
                "POST", SEARCH_URL, headers=self.headers, json=query_dict
            )
            data = res.json()
            for result in data["results"]:
                page_id = result["id"]
                page_ids.append(page_id)

            if data["next_cursor"] is None:
                done = True
                break
            else:
                next_cursor = data["next_cursor"]
        return page_ids

    def load_data(
        self,
        page_ids: List[str] = [],
        database_ids: Optional[List[str]] = None,
        load_all_if_empty: bool = False,
    ) -> List[Document]:
        """
        Load data from the input directory.

        Args:
            page_ids (List[str]): List of page ids to load.
            database_ids Optional (List[str]): List database ids from which to load page ids.
            load_all_if_empty (bool): If True, load all pages and dbs if no page_ids or database_ids are provided.

        Returns:
            List[Document]: List of documents.

        """
        if not page_ids and not database_ids:
            if not load_all_if_empty:
                raise ValueError(
                    "Must specify either `page_ids` or `database_ids` if "
                    "`load_all_if_empty` is False."
                )
            else:
                database_ids = self.list_databases()
                page_ids = self.list_pages()

        docs = []
        all_page_ids = set(page_ids) if page_ids is not None else set()
        # TODO: in the future add special logic for database_ids
        if database_ids is not None:
            for database_id in database_ids:
                # get all the pages in the database
                db_page_ids = self.query_database(database_id)
                all_page_ids.update(db_page_ids)

        for page_id in all_page_ids:
            page_text = self.read_page(page_id)
            docs.append(
                Document(text=page_text, id_=page_id, extra_info={"page_id": page_id})
            )

        return docs

    def list_databases(self) -> List[str]:
        """List all databases in the Notion workspace."""
        query_dict = {"filter": {"property": "object", "value": "database"}}
        res = self._request_with_retry(
            "POST", SEARCH_URL, headers=self.headers, json=query_dict
        )
        res.raise_for_status()
        data = res.json()
        return [db["id"] for db in data["results"]]

    def list_pages(self) -> List[str]:
        """List all pages in the Notion workspace."""
        query_dict = {"filter": {"property": "object", "value": "page"}}
        res = self._request_with_retry(
            "POST", SEARCH_URL, headers=self.headers, json=query_dict
        )
        res.raise_for_status()
        data = res.json()
        return [page["id"] for page in data["results"]]


if __name__ == "__main__":
    reader = NotionPageReader()
    print(reader.search("What I"))

    # get list of database from notion
    databases = reader.list_databases()
