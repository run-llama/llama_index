"""Notion reader."""

import os
import time
from typing import Any, Dict, List, Optional, Callable
import requests  # type: ignore
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document
from typing import NewType, Iterable
from notion_client.helpers import iterate_paginated_api, collect_paginated_api


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
# TODO page_id_t | str
# TODO try to get rid of if start_cursor is not None:
# TODO page titles???

# -------------------------------------------------
# Notes


# This code has two types of databases
# 1. Notion as a Database
# 2. Notion databases https://www.notion.com/help/intro-to-databases
# make sure not to mix them up


# What is the start_cursor?
#
#   A string that can be used to retrieve the next page of results
#   by passing the value as the start_cursor parameter to the same endpoint.
#   Only available when has_more is true.
#   ie if you have read up to page 10, your start cursor will be page 11
#
#   https://developers.notion.com/reference/intro#pagination
#
#   you should try not to have any custom code to handle pagination, just use library code
#   you may get an error related to start_cursor if you have a previous request fail

# -------------------------------------------------


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

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # META FUNCTIONS

    def _request_block(self, block_id: str, query_dict: json_t = {}) -> json_t:
        # AI: Helper function to get block data
        block_url = BLOCK_CHILD_URL_TMPL.format(block_id=block_id)
        res = self._request_with_retry(
            "GET",
            block_url,
            headers=self.headers,
            json=query_dict,
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

        def get_block_next_page(start_cursor: Optional[str]) -> json_t:
            self._print("_read_block get page")
            query_dict = {}
            if start_cursor is not None:
                query_dict["start_cursor"] = start_cursor

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

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # OTHER FUNCTIONS

    def get_all_pages_from_database(
        self, database_id: notion_db_id_t, query_dict: Dict[str, Any]
    ) -> list[json_t]:
        """Get all pages from a database using pagination."""
        return collect_paginated_api(self._request_database, database_id=database_id)

    def get_all_page_ids_from_database(
        self,
        database_id: notion_db_id_t,
        query_dict: Dict[str, Any] = {"page_size": 100},
    ) -> list[page_id_t]:
        """Get all page ids from a database using pagination."""
        pages = self.get_all_pages_from_database(database_id, query_dict)
        return [page["id"] for page in pages]

    def query_database(  # bad function name
        self,
        database_id: notion_db_id_t,
        query_dict: Dict[str, Any] = {"page_size": 100},
    ) -> List[page_id_t]:
        """Get all the pages from a Notion database."""
        return self.get_all_page_ids_from_database(database_id, query_dict)

    def search(self, query: str) -> List[str]:
        """Search Notion page given a text query."""

        def search_pages(start_cursor: Optional[str]) -> json_t:
            self._print("search_pages -- getting new data")

            query_dict = {"query": query}
            if start_cursor is not None:
                query_dict["start_cursor"] = start_cursor
            return self._request_search(query_dict)

        results = iterate_paginated_api(search_pages)
        return [result["id"] for result in results]

    def get_page_ids_from_databases(
        self, database_id_list: list[notion_db_id_t]
    ) -> set[page_id_t]:
        page_ids: set[page_id_t] = set()
        for database_id in database_id_list:
            page_ids.update(self.get_all_page_ids_from_database(database_id))

        assert set(page_ids).issubset(self.get_all_pages())
        return page_ids

    # TODO compare this to get_all_pages_and_databases
    def load_data(
        self,
        page_ids: List[page_id_t] = [],
        database_ids: List[notion_db_id_t] = [],
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

        database_page_ids: set[page_id_t] = self.get_page_ids_from_databases(
            database_ids
        )
        all_page_ids: set[page_id_t] = database_page_ids.union(set(page_ids))

        docs: list[Document] = []
        docs.extend(self.get_notion_databases(databases=database_ids))
        docs.extend(self.get_pages(pages=all_page_ids))

        return docs

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # META FUNCTIONS

    def _list_ids(self, function_name: str, value: str) -> List[str]:
        """List all databases in the Notion workspace."""
        # TODO use search function? how does query differ from filter?

        def search_databases(start_cursor: Optional[str]) -> json_t:
            self._print(f"{function_name} -- getting new data")
            query_dict: json_t = {"filter": {"property": "object", "value": value}}
            if start_cursor is not None:
                query_dict["start_cursor"] = start_cursor

            return self._request_search(query_dict)

        results = iterate_paginated_api(search_databases)
        ids: list[str] = [res["id"] for res in results]
        assert len(ids) == len(set(ids))
        self._print(f"found {len(ids)} databases")
        return ids

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # NOTION DATABASES INTERNAL

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

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # NOTION DATABASES

    def list_database_ids(self) -> List[notion_db_id_t]:
        """List all databases in the Notion workspace."""
        return [
            notion_db_id_t(id) for id in self._list_ids("list_database_ids", "database")
        ]

    def get_notion_database_text(
        self,
        database_id: notion_db_id_t,
        format_db_json: format_json_f = default_format_db_json,
    ) -> str:
        """Read a database."""
        self._print(f"reading database {database_id}")
        database_data = self._request_database(database_id, {})

        return format_db_json(database_data)

    def get_notion_database(
        self,
        database_id: notion_db_id_t,
        format_db_json: format_json_f = default_format_db_json,
    ) -> Document:
        """Get a database in the Notion workspace."""
        database_text = self.get_notion_database_text(
            database_id, format_db_json=format_db_json
        )
        return Document(
            text=database_text, id_=database_id, extra_info={"database_id": database_id}
        )

    def get_notion_databases(
        self,
        databases: List[notion_db_id_t],
        format_db_json: format_json_f = default_format_db_json,
    ) -> List[Document]:
        """Get all databases in the Notion workspace."""
        return [
            self.get_notion_database(database_id, format_db_json=format_db_json)
            for database_id in databases
        ]

    def get_all_notion_databases(
        self,
        format_db_json: format_json_f = default_format_db_json,
    ) -> List[Document]:
        """Get all databases in the Notion workspace."""
        databases = self.list_database_ids()
        return self.get_notion_databases(
            databases=databases, format_db_json=format_db_json
        )

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # PAGES

    def list_page_ids(self) -> List[page_id_t]:
        """List all pages in the Notion workspace."""
        return [page_id_t(id) for id in self._list_ids("list_page_ids", "page")]

    def read_page_text(self, page_id: page_id_t) -> str:
        """Read a page."""
        self._print(f"reading page {page_id}")
        return self._read_block(page_id)

    def get_page(self, page_id: page_id_t) -> Document:
        """Get a page in the Notion workspace."""
        page_text = self.read_page_text(page_id)
        return Document(text=page_text, id_=page_id, extra_info={"page_id": page_id})

    def get_pages(self, pages: Iterable[page_id_t]) -> List[Document]:
        """Get all pages in the Notion workspace."""
        return [self.get_page(page_id) for page_id in pages]

    def get_all_pages(
        self,
    ) -> List[Document]:
        """Get all pages in the Notion workspace."""
        pages = self.list_page_ids()
        return self.get_pages(pages=pages)

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # CONVENIENCE FUNCTIONS

    def get_all_pages_and_databases(
        self,
        format_db_json: format_json_f = default_format_db_json,
    ) -> List[Document]:
        return (
            self.get_all_notion_databases(format_db_json=format_db_json)
            + self.get_all_pages()
        )

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # HELPER FUNCTIONS

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


"""

 ERRORS





why is this invalid: Traceback (most recent call last):
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/llama_index/readers/notion/base.py", line 184, in _request_with_retry
    response.raise_for_status()
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/requests/models.py", line 1024, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 400 Client Error: Bad Request for url: https://api.notion.com/v1/blocks/149c2e93-a412-80eb-af36-f334f97f1b93/children

This is because i do not have an integration to the execs notion,
i need to figure out how to avoid what i don't have access to







Traceback (most recent call last):
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/llama_index/readers/notion/base.py", line 185, in _request_with_retry
    response.raise_for_status()
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/requests/models.py", line 1024, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 400 Client Error: Bad Request for url: https://api.notion.com/v1/blocks/149c2e93-a412-80eb-af36-f334f97f1b93/children

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/henry/Documents/Documents - MacBook Pro (6)/Git.nosync/DatabaseAware/test.py", line 41, in <module>
    test_notion_reader()
  File "/Users/henry/Documents/Documents - MacBook Pro (6)/Git.nosync/DatabaseAware/test.py", line 22, in test_notion_reader
    notion_reader.get_page(page_id=page_id_t("149c2e93-a412-80eb-af36-f334f97f1b93"))
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/llama_index/readers/notion/base.py", line 422, in get_page
    page_text = self.read_page_text(page_id)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/llama_index/readers/notion/base.py", line 418, in read_page_text
    return self._read_block(page_id)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/llama_index/readers/notion/base.py", line 144, in _read_block
    for result in iterate_paginated_api(get_block_next_page):
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/notion_client/helpers.py", line 44, in iterate_paginated_api
    response = function(**kwargs, start_cursor=next_cursor)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/llama_index/readers/notion/base.py", line 141, in get_block_next_page
    return self._request_block(block_id, query_dict)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/llama_index/readers/notion/base.py", line 104, in _request_block
    res = self._request_with_retry(
          ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/llama_index/readers/notion/base.py", line 192, in _request_with_retry
    raise requests.exceptions.HTTPError(
requests.exceptions.HTTPError: Request failed: {"object":"error","status":400,"code":"validation_error","message":"body failed validation: body.start_cursor should be not present, instead was `\"149c2e93-a412-8069-b844-cf54d8844af9\"`.","request_id":"bf7126a0-befd-4c8a-922c-a57c4da292fd"}
henry@MacBook-Pro-46 DatabaseAware %









"""
