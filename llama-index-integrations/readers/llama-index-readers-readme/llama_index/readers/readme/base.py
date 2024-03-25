"""Readme reader."""

import requests
import base64
import math
from typing import List

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class ReadmeReader(BaseReader):
    """Readme reader. Reads data from a Readme.com docs.

    Args:
        api_key (str): Readme.com API Key
    """

    def __init__(self, api_key: str) -> None:
        """Initialize Readme reader."""
        self.api_key = base64.b64encode(bytes(f"{api_key}:", "utf-8")).decode("utf-8")
        self._headers = {
            "accept": "*/*",
            "authorization": f"Basic {self.api_key}",
            "Content-Type": "application/json",
        }

    def load_data(self) -> List[Document]:
        """Load data from the docs (pages).

        Returns:
            List[Document]: List of documents.
        """
        from bs4 import BeautifulSoup

        results = []

        docs = self.get_all_docs()
        for doc in docs:
            body = doc["body_html"]
            if body is None:
                continue
            soup = BeautifulSoup(body, "html.parser")
            body = soup.get_text()
            extra_info = {
                "id": doc["id"],
                "title": doc["title"],
                "type": doc["title"],
                "slug": doc["slug"],
                "updated_at": doc["updatedAt"],
            }

            results.append(
                Document(
                    text=body,
                    extra_info=extra_info,
                )
            )

        return results

    def get_all_docs(self):
        """
        Retrieves all documents, along with their information, categorized by categories.

        Returns:
            list: A list containing dictionaries with document information.
        """
        categories = self.get_all_categories()
        docs = []
        for category in categories:
            category_docs = self.get_docs_in_category(category.get("slug"))
            documents_slugs = [
                category_doc.get("slug") for category_doc in category_docs
            ]
            for document_slug in documents_slugs:
                doc = self.get_document_info(document_slug)
                doc["category_name"] = category["title"]
                docs.append(doc)

        return docs

    def get_docs_in_category(self, category_slug):
        """
        Retrieves documents belonging to a specific category.

        Args:
            category_slug (str): The slug of the category.

        Returns:
            list: A list containing dictionaries with document information.
        """
        url = f"https://dash.readme.com/api/v1/categories/{category_slug}/docs"
        response = requests.get(url, headers=self._headers)

        docs = response.json()

        # Filter documents hidden=False
        return [doc for doc in docs if not doc.get("hidden", True)]

    def get_document_info(self, document_slug):
        """
        Retrieves information about a specific document.

        Args:
            document_slug (str): The slug of the document.

        Returns:
            dict: A dictionary containing document information.
        """
        url = f"https://dash.readme.com/api/v1/docs/{document_slug}"
        response = requests.get(url, headers=self._headers)

        return response.json()

    def get_categories_page(self, params, page):
        """
        Sends a GET request to a specific page of categories.

        Args:
            params (dict): Parameters of the request, such as perPage and others.
            page (int): The number of the page to be retrieved.

        Returns:
            tuple: A tuple containing the total number of items and the retrieved categories.
        """
        url = "https://dash.readme.com/api/v1/categories"
        params["page"] = page
        response = requests.get(url, params=params, headers=self._headers)
        # total counts and categories
        return int(response.headers.get("x-total-count", 0)), response.json()

    def get_all_categories(self):
        """
        Retrieves all categories from the API.

        Returns:
            list: A list containing all categories with type "guide".
        """
        perPage = 100
        page = 1
        params = {
            "perPage": perPage,
            "page": page,
        }

        total_count, categories = self.get_categories_page(params=params, page=1)
        remaining_pages = math.ceil(total_count / perPage) - 1

        for i in range(2, remaining_pages + 2):
            categories.extend(self.get_categories_page(params=params, page=i))

        # Include just categories with type: "guide"
        return [category for category in categories if category.get("type") == "guide"]
