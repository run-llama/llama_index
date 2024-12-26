"""Notion reader."""

import os
import time
from typing import Any, Dict, List, Optional, Callable
import requests  # type: ignore
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document
from typing import NewType
from notion_client.helpers import iterate_paginated_api


INTEGRATION_TOKEN_NAME = "NOTION_INTEGRATION_TOKEN"
BLOCK_CHILD_URL_TMPL = "https://api.notion.com/v1/blocks/{block_id}/children"
DATABASE_URL_TMPL = "https://api.notion.com/v1/databases/{database_id}/query"
SEARCH_URL = "https://api.notion.com/v1/search"

NOTION_READER_PRINT_PREFIX = "Notion reader : "

json_t = dict[str, Any]
format_json_f = Callable[[json_t], str]
page_id_t = NewType("page_id_t", str)
notion_db_id_t = NewType("notion_db_id_t", str)


# TODO may be a good idea to use https://github.com/ramnes/notion-sdk-py
# TODO get titles from databases


# This code has two types of databases
# 1. Notion as a Database
# 2. Notion databases https://www.notion.com/help/intro-to-databases
# make sure not to mix them up


class NotionPageReader(BasePydanticReader):
    """Notion Page reader.

    Reads a set of Notion pages.

    Args:
        integration_token (str): Notion integration token.
        print_feedback (bool): Whether to print feedback during operations.

    """

    is_remote: bool = True
    token: str
    headers: Dict[str, str]
    print_feedback: bool = False

    def __init__(
        self, integration_token: Optional[str] = None, print_feedback: bool = False
    ) -> None:
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
        self.print_feedback = print_feedback

    @classmethod
    def class_name(cls) -> str:
        """Get the name identifier of the class."""
        return "NotionPageReader"

    def _request_block(self, block_id: str, query_dict: json_t = {}) -> json_t:
        # AI: Helper function to get block data
        block_url = BLOCK_CHILD_URL_TMPL.format(block_id=block_id)
        res = self._request_with_retry(
            "GET", block_url, headers=self.headers, json=query_dict
        )
        return res.json()

    def _request_database(
        self, database_id: notion_db_id_t, query_dict: json_t
    ) -> json_t:
        # AI: Helper function to query database
        res = self._request_with_retry(
            "POST",
            DATABASE_URL_TMPL.format(database_id=database_id),
            headers=self.headers,
            json=query_dict,
        )
        return res.json()

    def _request_search(self, query_dict: json_t) -> json_t:
        # AI: Helper function for search endpoint
        res = self._request_with_retry(
            "POST", SEARCH_URL, headers=self.headers, json=query_dict
        )
        return res.json()

    def _read_block(self, block_id: str, num_tabs: int = 0) -> str:
        """Read a block."""
        block_text: str = "\n"

        def get_block_next_page(**kwargs: Any) -> json_t:
            # AI: Only include kwargs if they have a valid start_cursor
            self._print("_read_block get page")

            query_dict = {}
            if "start_cursor" in kwargs and kwargs["start_cursor"] is not None:
                query_dict["start_cursor"] = kwargs["start_cursor"]
            return self._request_block(block_id, query_dict)

        # Iterate through all block results using the paginated API helper
        for result in iterate_paginated_api(get_block_next_page):
            result_type: str = result["type"]
            result_obj: json_t = result[result_type]

            if "rich_text" in result_obj:
                for rich_text in result_obj["rich_text"]:
                    # skip if doesn't have text object
                    if "text" in rich_text:
                        text: str = rich_text["text"]["content"]
                        prefix: str = "\t" * num_tabs
                        block_text += prefix + text + "\n"

            result_block_id: str = result["id"]
            has_children: bool = result["has_children"]
            if has_children:
                children_text: str = self._read_block(
                    result_block_id, num_tabs=num_tabs + 1
                )
                block_text += children_text + "\n"
            block_text += "\n"

        return block_text

    def _request_with_retry(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        json: Optional[Dict[str, Any]] = None,
    ) -> requests.Response:
        """Make a request with retry and rate limit handling."""
        max_retries = 5
        backoff_factor = 1

        RATE_LIMIT_ERROR_CODE: int = 429

        for attempt in range(max_retries):
            try:
                response: requests.Response = requests.request(
                    method, url, headers=headers, json=json
                )
                response.raise_for_status()
                return response
            except requests.exceptions.HTTPError:
                if response.status_code == RATE_LIMIT_ERROR_CODE:
                    retry_after = int(response.headers.get("Retry-After", 1))
                    time.sleep(backoff_factor * (2**attempt) + retry_after)
                else:
                    raise requests.exceptions.HTTPError(
                        f"Request failed: {response.text}"
                    )
            except requests.exceptions.RequestException as err:
                raise requests.exceptions.RequestException(f"Request failed: {err}")
        raise Exception("Maximum retries exceeded")

    def read_page(self, page_id: page_id_t) -> str:
        """Read a page."""
        self._print(f"reading page {page_id}")
        return self._read_block(page_id)

    def get_all_pages_from_database(
        self, database_id: notion_db_id_t, query_dict: Dict[str, Any]
    ) -> list[json_t]:
        """Get all pages from a database using pagination."""

        # AI: Using iterate_paginated_api to handle pagination
        def query_database(**kwargs: Any) -> json_t:
            return self._request_database(database_id, kwargs)

        return list(iterate_paginated_api(query_database, **query_dict))

    # TODO this function name can be misleading, it does not say it will return page ids in the signature
    # TODO this function name is not very descriptive
    # TODO page_ids_from_notion_database
    def query_database(
        self,
        database_id: notion_db_id_t,
        query_dict: Dict[str, Any] = {"page_size": 100},
    ) -> List[page_id_t]:
        """Get all the pages from a Notion database."""
        pages = self.get_all_pages_from_database(database_id, query_dict)
        return [page["id"] for page in pages]

    def search(self, query: str) -> List[str]:
        """Search Notion page given a text query."""

        # AI: Using iterate_paginated_api with proper cursor handling
        def search_pages(**kwargs: Any) -> json_t:
            self._print("search_pages get page")

            search_params = {"query": query}
            # AI: Only include start_cursor if it's provided and not None
            if "start_cursor" in kwargs and kwargs["start_cursor"] is not None:
                search_params["start_cursor"] = kwargs["start_cursor"]
            return self._request_search(search_params)

        results = iterate_paginated_api(search_pages)
        return [result["id"] for result in results]

    # TODO this name is bad
    def load_data(
        self,
        page_ids: List[page_id_t] = [],
        database_ids: Optional[
            List[notion_db_id_t]
        ] = None,  # please note : this does not extract any useful table data, only children pages
        load_all_if_empty: bool = False,
    ) -> List[Document]:
        """Load data from the input directory.

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
                database_ids = self.list_database_ids()
                page_ids = self.list_page_ids()

        docs: list[Document] = []
        all_page_ids: set[str] = set(page_ids)
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

    @staticmethod
    def default_format_db_json(json_database: json_t) -> str:
        # TODO get title of the database

        database_text: str = "\n Notion Database Start  -----------------\n"
        for row in json_database.get("results", []):
            properties: json_t = row.get("properties", {})

            database_text += "\nNew row\n"
            for prop_name, prop_value in properties.items():
                prop_value: json_t = prop_value

                # this logic remove useless metadata and makes the table human readable compared to json
                from_type: Any = prop_value.get(prop_value["type"], [])
                database_text += prop_name + ",type:" + prop_value["type"] + ",data:"

                # allow most customization in future, although error checking can be difficult
                if prop_value["type"] == "relation":
                    database_text += str(from_type)
                elif prop_value["type"] == "multi_select":
                    database_text += str(from_type)
                elif prop_value["type"] == "rich_text":
                    database_text += str(from_type)
                elif prop_value["type"] == "title":
                    database_text += str(from_type)
                elif prop_value["type"] == "checkbox":
                    database_text += str(from_type)
                elif prop_value["type"] == "url":
                    database_text += str(from_type)
                elif prop_value["type"] == "text":
                    database_text += str(from_type)
                elif prop_value["type"] == "email":
                    database_text += str(from_type)
                else:
                    # print("Unknown type: ", prop_value["type"])
                    database_text += str(prop_value)

                # database_text += "  ORIGINAL "+ str(prop_value) # use this line to see what data is being filtered
                database_text += "\n"

        return database_text + "\nNotion Database End  -----------------\n"

    def read_notion_database(
        self,
        database_id: notion_db_id_t,
        format_db_json: format_json_f = default_format_db_json,
    ) -> str:
        """Read a database."""
        self._print(f"reading database {database_id}")

        # https://developers.notion.com/reference/post-database-query
        database_data = self._request_database(database_id, {})

        return format_db_json(database_data)

    def get_all_databases(
        self,
        format_db_json: format_json_f = default_format_db_json,
    ) -> List[Document]:
        """Get all databases in the Notion workspace."""
        databases = self.list_database_ids()
        docs: list[Document] = []
        for database_id in databases:
            database_text = self.read_notion_database(
                database_id, format_db_json=format_db_json
            )
            doc = Document(
                text=database_text,
                id_=database_id,
                extra_info={"database_id": database_id},
            )
            docs.append(doc)

        return docs

    def list_database_ids(self) -> List[notion_db_id_t]:
        """List all databases in the Notion workspace."""

        def search_databases(**kwargs: Any) -> json_t:
            self._print("list_database_ids -- getting new data")

            search_params = {"filter": {"property": "object", "value": "database"}}
            if "start_cursor" in kwargs and kwargs["start_cursor"] is not None:
                search_params["start_cursor"] = kwargs["start_cursor"]
            return self._request_search(search_params)

        results = iterate_paginated_api(search_databases)
        s = {db["id"] for db in results}

        self._print(f"found {len(s)} databases")

        return list(s)

    def list_page_ids(self) -> List[page_id_t]:
        """List all pages in the Notion workspace."""

        def search_pages(**kwargs: Any) -> json_t:
            self._print("list_page_ids -- getting new data")

            search_params = {"filter": {"property": "object", "value": "page"}}
            if "start_cursor" in kwargs and kwargs["start_cursor"] is not None:
                search_params["start_cursor"] = kwargs["start_cursor"]
            return self._request_search(search_params)

        results = iterate_paginated_api(search_pages)
        s = {page["id"] for page in results}

        self._print(f"found {len(s)} pages")

        return list(s)

    def get_all_pages(
        self,
        format_db_json: format_json_f = default_format_db_json,
    ) -> List[Document]:
        """Get all pages in the Notion workspace."""
        pages = self.list_page_ids()
        docs: list[Document] = []
        for page_id in pages:
            page_text = self.read_page(page_id)
            doc = Document(text=page_text, id_=page_id, extra_info={"page_id": page_id})
            docs.append(doc)

        return docs

    def get_all_pages_and_databases(
        self,
        format_db_json: format_json_f = default_format_db_json,
    ) -> List[Document]:
        """Get all pages and databases in the Notion workspace."""
        return self.get_all_databases(
            format_db_json=format_db_json,
        ) + self.get_all_pages(
            format_db_json=format_db_json,
        )

    def _print(self, message: str) -> None:
        """Helper method to print feedback messages if print_feedback is enabled.

        Args:
            message (str): The message to print
        """
        if self.print_feedback:
            print(NOTION_READER_PRINT_PREFIX + message)


if __name__ == "__main__":
    reader = NotionPageReader()
    print(reader.search("What I"))

    # get list of database from notion
    databases = reader.list_database_ids()
