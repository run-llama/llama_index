import time
from typing import List

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class WholeSiteReader(BaseReader):
    """
    BFS Web Scraper for websites.

    This class provides functionality to scrape entire websites using a breadth-first search algorithm.
    It navigates web pages from a given base URL, following links that match a specified prefix.

    Attributes:
        prefix (str): URL prefix to focus the scraping.
        max_depth (int): Maximum depth for BFS algorithm.

    Args:
        prefix (str): URL prefix for scraping.
        max_depth (int, optional): Maximum depth for BFS. Defaults to 10.
    """

    def __init__(self, prefix: str, max_depth: int = 10) -> None:
        """
        Initialize the WholeSiteReader with the provided prefix and maximum depth.
        """
        self.prefix = prefix
        self.max_depth = max_depth
        self.driver = self.setup_driver()

    def setup_driver(self):
        """
        Sets up the Selenium WebDriver for Chrome.

        Returns:
            WebDriver: An instance of Chrome WebDriver.
        """
        try:
            import chromedriver_autoinstaller
        except ImportError:
            raise ImportError("Please install chromedriver_autoinstaller")

        opt = webdriver.ChromeOptions()
        opt.add_argument("--start-maximized")
        chromedriver_autoinstaller.install()
        return webdriver.Chrome(options=opt)

    def clean_url(self, url):
        return url.split("#")[0]

    def restart_driver(self):
        self.driver.quit()
        self.driver = self.setup_driver()

    def extract_content(self):
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        body_element = self.driver.find_element(By.TAG_NAME, "body")
        return body_element.text.strip()

    def extract_links(self):
        js_script = """
            var links = [];
            var elements = document.getElementsByTagName('a');
            for (var i = 0; i < elements.length; i++) {
                var href = elements[i].href;
                if (href) {
                    links.push(href);
                }
            }
            return links;
            """
        return self.driver.execute_script(js_script)

    def load_data(self, base_url: str) -> List[Document]:
        """Load data from the base URL using BFS algorithm.

        Args:
            base_url (str): Base URL to start scraping.


        Returns:
            List[Document]: List of scraped documents.
        """
        added_urls = set()
        urls_to_visit = [(base_url, 0)]
        documents = []

        while urls_to_visit:
            current_url, depth = urls_to_visit.pop(0)

            print(f"Visiting: {current_url}, {len(urls_to_visit)} left")

            if depth > self.max_depth:
                continue
            try:
                self.driver.get(current_url)
                page_content = self.extract_content()

                # links = self.driver.find_elements(By.TAG_NAME, 'a')
                links = self.extract_links()
                # clean all urls
                links = [self.clean_url(link) for link in links]
                # extract new links
                links = [link for link in links if link not in added_urls]
                print(f"Found {len(links)} new potential links")
                for href in links:
                    try:
                        if href.startswith(self.prefix) and href not in added_urls:
                            urls_to_visit.append((href, depth + 1))
                            added_urls.add(href)
                    except Exception:
                        continue

                documents.append(
                    Document(text=page_content, extra_info={"URL": current_url})
                )
                time.sleep(1)

            except WebDriverException:
                print("WebDriverException encountered, restarting driver...")
                self.restart_driver()
            except Exception as e:
                print(f"An unexpected exception occurred: {e}, skipping URL...")
                continue

        self.driver.quit()
        return documents
