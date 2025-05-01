"""MangoppsGuides reader."""

import re
from typing import List
from urllib.parse import urlparse

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class MangoppsGuidesReader(BaseReader):
    """
    MangoppsGuides reader. Reads data from a MangoppsGuides workspace.

    Args:
        domain_url (str): MangoppsGuides domain url
        limir (int): depth to crawl

    """

    def __init__(self) -> None:
        """Initialize MangoppsGuides reader."""

    def load_data(self, domain_url: str, limit: int) -> List[Document]:
        """
        Load data from the workspace.

        Returns:
            List[Document]: List of documents.

        """
        import requests
        from bs4 import BeautifulSoup

        self.domain_url = domain_url
        self.limit = limit
        self.start_url = f"{self.domain_url}/home/"

        fetched_urls = self.crawl_urls()[: self.limit]

        results = []

        guides_pages = {}
        for url in fetched_urls:
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.content, "html.parser")

                page_title = soup.find("title").text

                # Remove the div with aria-label="Table of contents"
                table_of_contents_div = soup.find(
                    "div", {"aria-label": "Table of contents"}
                )
                if table_of_contents_div:
                    table_of_contents_div.decompose()

                # Remove header and footer
                header = soup.find("header")
                if header:
                    header.decompose()
                footer = soup.find("footer")
                if footer:
                    footer.decompose()

                # Exclude links and their text content from the main content
                for link in soup.find_all("a"):
                    link.decompose()

                # Remove empty elements from the main content
                for element in soup.find_all():
                    if element.get_text(strip=True) == "":
                        element.decompose()

                # Find the main element containing the desired content
                main_element = soup.find(
                    "main"
                )  # Replace "main" with the appropriate element tag or CSS class

                # Extract the text content from the main element
                if main_element:
                    text_content = main_element.get_text("\n")
                    # Remove multiple consecutive newlines and keep only one newline
                    text_content = re.sub(r"\n+", "\n", text_content)
                else:
                    text_content = ""

                page_text = text_content

                guides_page = {}
                guides_page["title"] = page_title
                guides_page["text"] = page_text
                guides_pages[url] = guides_page
            except Exception as e:
                print(f"Failed for {url} => {e}")

        for k, v in guides_pages.items():
            extra_info = {"url": k, "title": v["title"]}
            results.append(
                Document(
                    text=v["text"],
                    extra_info=extra_info,
                )
            )

        return results

    def crawl_urls(self) -> List[str]:
        """Crawls all the urls from given domain."""
        self.visited = []

        fetched_urls = self.fetch_url(self.start_url)
        return list(set(fetched_urls))

    def fetch_url(self, url):
        """Fetch the urls from given domain."""
        import requests
        from bs4 import BeautifulSoup

        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        self.visited.append(url)

        newurls = []
        for link in soup.find_all("a"):
            href: str = link.get("href")
            if href and urlparse(href).netloc == self.domain_url:
                newurls.append(href)
            elif href and href.startswith("/"):
                newurls.append(f"{self.domain_url}{href}")

        for newurl in newurls:
            if (
                newurl not in self.visited
                and not newurl.startswith("#")
                and f"https://{urlparse(newurl).netloc}" == self.domain_url
                and len(self.visited) <= self.limit
            ):
                newurls = newurls + self.fetch_url(newurl)

        return list(set(newurls))


if __name__ == "__main__":
    reader = MangoppsGuidesReader()
    print("Initialized MangoppsGuidesReader")
    output = reader.load_data(domain_url="https://guides.mangoapps.com", limit=5)
    print(output)
