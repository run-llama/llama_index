import requests
from typing import List, Dict

DEFAULT_GITBOOK_API_URL = "https://api.gitbook.com/v1"


class GitbookClient:
    """Gitbook Restful API Client.

    Helper Class to invoke gitbook restful api & parse result

    Args:
        api_token (str): Gitbook API Token.
        api_url (str): Gitbook API Endpoint.
    """

    def __init__(self, api_token: str, api_url: str = DEFAULT_GITBOOK_API_URL):
        self.api_token = api_token
        self.base_url = api_url or DEFAULT_GITBOOK_API_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    def _make_request(self, url: str) -> Dict:
        """Helper method to handle common HTTP GET requests."""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return self._handle_error(e)

    def get_space(self, space_id) -> Dict:
        """Gets information for a specific space."""
        url = f"{self.base_url}/spaces/{space_id}"
        return self._make_request(url)

    def list_pages(self, space_id) -> List[Dict]:
        """Gets all pages in a specific space."""
        space_info = self.get_space(space_id)
        url = f"{self.base_url}/spaces/{space_id}/content"
        space = self._make_request(url)

        pages_info = []
        for page in space.get("pages"):
            GitbookClient._extract_page_info(
                pages_info, page, space_info.get("title", "ROOT")
            )
        return pages_info

    def get_page(self, space_id, page_id) -> Dict:
        """Gets the details of a specific page."""
        url = (
            f"{self.base_url}/spaces/{space_id}/content/page/{page_id}?format=markdown"
        )
        return self._make_request(url)

    def get_page_markdown(self, space_id, page_id) -> str:
        """Gets the content of a specific page in Markdown format."""
        page_content = self.get_page(space_id, page_id)
        return page_content.get("markdown")

    def _handle_error(self, response):
        """Handles HTTP errors."""
        if isinstance(response, requests.exceptions.HTTPError):
            error_message = f"Error: {response.response.status_code} Client Error: {response.response.reason}"
        else:
            error_message = f"Error: {response}"
        raise Exception(error_message)

    @staticmethod
    def _extract_page_info(
        pages: list, page: dict, prev_title: str = "", parent: str = ""
    ):
        pageType = page.get("type", "")
        title = prev_title + " > " + page.get("title")
        id = page.get("id")
        if pageType == "document":
            pages.append(
                {
                    "id": id,
                    "title": title,
                    "path": page.get("path"),
                    "description": page.get("description", ""),
                    "parent": parent,
                }
            )
            for _page in page.get("pages"):
                GitbookClient._extract_page_info(pages, _page, title, id)
        elif pageType == "group":
            for _page in page.get("pages"):
                GitbookClient._extract_page_info(pages, _page, title, id)
