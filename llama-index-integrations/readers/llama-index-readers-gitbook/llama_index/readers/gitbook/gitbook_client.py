import requests
from typing import Optional

DEFAULT_GITBOOK_API_URL = "https://api.gitbook.com/v1"

class GitbookClient:
    """Gitbook Restful API Client.

    Helper Class to invoke gitbook restful api & parse result

    Args:
        api_token (str): Gitbook API Token.
        api_url (str): Gitbook API Endpoint.
    """    

    def __init__(self, api_token:str, api_url:Optional[str] = None):
        self.api_token = api_token
        self.base_url = api_url or DEFAULT_GITBOOK_API_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

    def get_space(self, space_id)->dict:
        """Gets information for a specific space."""
        url = f"{self.base_url}/spaces/{space_id}"
        response = requests.get(url, headers=self.headers)
        return response.json() if response.status_code == 200 else self._handle_error(response)

    def list_pages(self, space_id)->list[dict]:
        """Gets all pages in a specific space."""
        space_info = self.get_space(space_id)
        url = f"{self.base_url}/spaces/{space_id}/content"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            space = response.json()
            pages_info=[]
            for page in space.get("pages"):
                GitbookClient._extract_page_info(pages_info, page, space_info.get("title","ROOT"))
            return pages_info            
        else:
            return self._handle_error(response)

    def get_page(self, space_id, page_id) -> dict:
        """Gets the details of a specific page."""
        url = f"{self.base_url}/spaces/{space_id}/content/page/{page_id}?format=markdown"
        response = requests.get(url, headers=self.headers)
        return response.json() if response.status_code == 200 else self._handle_error(response)

    def get_page_markdown(self, space_id, page_id) -> str:
        """Gets the content of a specific page in Markdown format."""
        page_content = self.get_page(space_id, page_id)
        return page_content.get("markdown")

    def _handle_error(self, response):
        """Methods for handling errors"""
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

    @staticmethod
    def _extract_page_info(pages:list, page:dict, prev_title:str = "", parent:str = "" ):
        pageType = page.get("type","")
        title = prev_title + " > " + page.get("title")
        id = page.get("id")
        if pageType == "document":
            pages.append({"id": id, "title": title, "path": page.get("path"), "description": page.get("description",""), "parent": parent})
            for _page in page.get("pages"):
                GitbookClient._extract_page_info(pages, _page, title, id)
        elif pageType == "group":
            for _page in page.get("pages"):
                GitbookClient._extract_page_info(pages, _page, title, id)
         