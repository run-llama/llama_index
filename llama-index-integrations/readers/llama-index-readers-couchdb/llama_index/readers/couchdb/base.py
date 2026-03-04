"""CouchDB client."""

import json
import logging
from typing import Dict, List, Optional

import couchdb3
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class SimpleCouchDBReader(BaseReader):
    """
    Simple CouchDB reader.

    Concatenates each CouchDB doc into Document used by LlamaIndex.

    Args:
        couchdb_url (str): CouchDB Full URL.
        max_docs (int): Maximum number of documents to load.

    """

    def __init__(
        self,
        user: str,
        pwd: str,
        host: str,
        port: int,
        couchdb_url: Optional[Dict] = None,
        max_docs: int = 1000,
    ) -> None:
        """Initialize with parameters."""
        if couchdb_url is not None:
            self.client = couchdb3.Server(couchdb_url)
        else:
            self.client = couchdb3.Server(f"http://{user}:{pwd}@{host}:{port}")
        self.max_docs = max_docs

    def load_data(self, db_name: str, query: Optional[str] = None) -> List[Document]:
        """
        Load data from the input directory.

        Args:
            db_name (str): name of the database.
            query (Optional[str]): query to filter documents.
                Defaults to None

        Returns:
            List[Document]: A list of documents.

        """
        documents = []
        db = self.client.get(db_name)
        if query is None:
            # if no query is specified, return all docs in database
            logging.debug("showing all docs")
            results = db.view("_all_docs", include_docs=True)
        else:
            logging.debug("executing query")
            results = db.find(query)

        if not isinstance(results, dict):
            logging.debug(results.rows)
        else:
            logging.debug(results)

        # check if more than one result
        if (
            not isinstance(results, dict)
            and hasattr(results, "rows")
            and results.rows is not None
        ):
            for row in results.rows:
                # check that the id field exists
                if "id" not in row:
                    raise ValueError("`id` field not found in CouchDB document.")
                documents.append(Document(text=json.dumps(row.doc)))
        else:
            # only one result
            if results.get("docs") is not None:
                for item in results.get("docs"):
                    # check that the _id field exists
                    if "_id" not in item:
                        raise ValueError("`_id` field not found in CouchDB document.")
                    documents.append(Document(text=json.dumps(item)))

        return documents
