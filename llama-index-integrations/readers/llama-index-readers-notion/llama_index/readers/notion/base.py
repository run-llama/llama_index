"""Notion reader."""

import os
import time
from typing import Any, Dict, List, Optional, Callable

import requests  # type: ignore
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

INTEGRATION_TOKEN_NAME = "NOTION_INTEGRATION_TOKEN"
BLOCK_CHILD_URL_TMPL = "https://api.notion.com/v1/blocks/{block_id}/children"
DATABASE_URL_TMPL = "https://api.notion.com/v1/databases/{database_id}/query"
SEARCH_URL = "https://api.notion.com/v1/search"

format_json_f = Callable[[dict], str] 




# TODO compare query_database vs load_data
# TODO get titles from databases
# TODO check you get all content from notion with manual tests
# TODO next_cursor need a unifying function to combine logic
# TODO maybe use a notion api wrapper


# This code has two types of databases
# 1. Notion as a Database
# 2. Notion databases https://www.notion.com/help/intro-to-databases
# make sure not to mix them up


class NotionPageReader(BasePydanticReader):
    """Notion Page reader.

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

    def _read_block(self, block_id: str, num_tabs: int = 0) -> str:
        """Read a block."""
        done = False
        result_lines_arr = []
        cur_block_id = block_id
        while not done:
            block_url = BLOCK_CHILD_URL_TMPL.format(block_id=cur_block_id)
            query_dict: Dict[str, Any] = {}

            res = self._request_with_retry(
                "GET", block_url, headers=self.headers, json=query_dict
            )
            data = res.json()

            for result in data["results"]:
                result_type = result["type"]
                result_obj = result[result_type]

                cur_result_text_arr = []
                if "rich_text" in result_obj:
                    for rich_text in result_obj["rich_text"]:
                        # skip if doesn't have text object
                        if "text" in rich_text:
                            text = rich_text["text"]["content"]
                            prefix = "\t" * num_tabs
                            cur_result_text_arr.append(prefix + text)

                result_block_id = result["id"]
                has_children = result["has_children"]
                if has_children:
                    children_text = self._read_block(
                        result_block_id, num_tabs=num_tabs + 1
                    )
                    cur_result_text_arr.append(children_text)

                cur_result_text = "\n".join(cur_result_text_arr)
                result_lines_arr.append(cur_result_text)

            if data["next_cursor"] is None:
                done = True
                break
            else:
                cur_block_id = data["next_cursor"]

        return "\n".join(result_lines_arr)

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

        for attempt in range(max_retries):
            try:
                response = requests.request(method, url, headers=headers, json=json)
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



    def get_all_pages_from_database(self, database_id: str, query_dict: Dict[str, Any]) -> list[dict]:

        pages : list[dict] = []

        # TODO a while True break / do while would work better here

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

        return pages


    # TODO this function name can be misleading, it does not say it will return page ids in the signature 
    # TODO this function name is not very descriptive
    # TODO page_ids_from_notion_database 
    def query_database(
        self, database_id: str, query_dict: Dict[str, Any] = {"page_size": 100}
    ) -> List[str]:
        """Get all the pages from a Notion database."""

        pages = self.get_all_pages_from_database(database_id, query_dict)
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

    # TODO this name is bad
    def load_data(
        self,
        page_ids: List[str] = [],
        database_ids: Optional[List[str]] = None, # please note : this does not extract any useful table data, only children pages
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

    @staticmethod
    def default_format_db_json(json_database: dict) -> str:
        
        # TODO get title of the database
    
        database_text = "\n Notion Database Start  -----------------\n"
        for row in json_database.get("results", []):
            properties = row.get("properties", {})

            database_text += "\nNew row\n"
            for prop_name, prop_value in properties.items():
                prop_value : dict = prop_value
                
                # this logic remove useless metadata and makes the table human readable compared to json
                from_type : any = prop_value.get(prop_value["type"], [])
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
                    #print("Unknown type: ", prop_value["type"])
                    database_text += str(prop_value)


                # database_text += "  ORIGINAL "+ str(prop_value) # use this line to see what data is being filtered
                database_text += "\n"

        return database_text + "\nNotion Database End  -----------------\n"
    

    def read_notion_database(self, database_id: str, format_db_json : format_json_f = default_format_db_json) -> str:

        """Read a database."""
        
        # https://developers.notion.com/reference/post-database-query
        database_data : dict = self._request_with_retry(
          "POST",  DATABASE_URL_TMPL.format(database_id=database_id), headers=self.headers, json={}
        ).json()

        return format_db_json(database_data)
    

    def get_all_databases(self, format_db_json : format_json_f = default_format_db_json, print_feedback : bool = False) -> List[Document]:
        """Get all databases in the Notion workspace."""

        databases = self.list_databases()
        if print_feedback:
            print("Found ", len(databases), " databases")

        docs : list[Document] = []
        for database_id in databases:
            
            if print_feedback:
                print("Reading database: ", database_id)

            database_text = self.read_notion_database(database_id, format_db_json=format_db_json)
            doc = Document(text=database_text, id_=database_id, extra_info={"database_id": database_id})
            docs.append(doc)

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
    

    def get_all_pages(self, format_db_json : format_json_f = default_format_db_json, print_feedback : bool = False) -> List[Document]:
        """Get all pages in the Notion workspace."""

        
        pages = self.list_pages()
        if print_feedback:
            # it's important for the user to know how long the operation will take
            print("Found ", len(pages), " pages")

        docs : list[Document] = []
        for page_id in pages:
            
            if print_feedback:
                print("Reading page: ", page_id)
            page_text = self.read_page(page_id)
            doc = Document(text=page_text, id_=page_id, extra_info={"page_id": page_id})
            docs.append(doc)

        return docs


    def get_all_pages_and_databases(self, format_db_json : format_json_f = default_format_db_json, print_feedback : bool = False) -> List[Document]:
        """Get all pages and databases in the Notion workspace."""

        return self.get_all_databases(format_db_json=format_db_json, print_feedback=print_feedback) + \
               self.get_all_pages(format_db_json=format_db_json, print_feedback=print_feedback)


if __name__ == "__main__":
    reader = NotionPageReader()
    print(reader.search("What I"))

    # get list of database from notion
    databases = reader.list_databases()
