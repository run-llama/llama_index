import logging
import time
import warnings
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

logger = logging.getLogger(__name__)


class WholeSiteReader(BaseReader):
    """
    BFS Web Scraper for websites.

    This class provides functionality to scrape entire websites using a breadth-first search algorithm.
    It navigates web pages from a given base URL, following links that match a specified prefix.

    Attributes:
        prefix (str): URL prefix to focus the scraping.
        max_depth (int): Maximum depth for BFS algorithm.
        delay (float): Delay in seconds between page requests.
        respect_robots_txt (bool): Whether to respect robots.txt rules.

    Args:
        prefix (str): URL prefix for scraping.
        max_depth (int, optional): Maximum depth for BFS. Defaults to 10.
        uri_as_id (bool, optional): Whether to use the URI as the document ID. Defaults to False.
        driver (Optional[webdriver.Chrome], optional): Custom Chrome WebDriver instance. Defaults to None.
        delay (float, optional): Delay in seconds between page requests. Defaults to 1.0.
        respect_robots_txt (bool, optional): Whether to respect robots.txt rules. Defaults to True.

    """

    def __init__(
        self,
        prefix: str,
        max_depth: int = 10,
        uri_as_id: bool = False,
        driver: Optional[webdriver.Chrome] = None,
        delay: float = 1.0,
        respect_robots_txt: bool = True,
    ) -> None:
        """
        Initialize the WholeSiteReader with the provided prefix and maximum depth.

        Args:
            prefix (str): URL prefix for scraping.
            max_depth (int): Maximum depth for BFS algorithm.
            uri_as_id (bool): Whether to use the URI as the document ID.
            driver (Optional[webdriver.Chrome]): Custom Chrome WebDriver instance.
            delay (float): Delay in seconds between page requests.
            respect_robots_txt (bool): Whether to respect robots.txt rules.

        """
        self.prefix = prefix
        self.max_depth = max_depth
        self.uri_as_id = uri_as_id
        self.driver = driver if driver else self.setup_driver()
        self.delay = delay
        self.respect_robots_txt = respect_robots_txt
        self._robot_parser: Optional[RobotFileParser] = None

        # Initialize robots.txt parser if enabled
        if self.respect_robots_txt:
            self._init_robot_parser()

    def _init_robot_parser(self) -> None:
        """
        Initialize the robots.txt parser for the base URL.

        This method fetches and parses the robots.txt file from the base URL
        to determine which URLs are allowed to be crawled.

        """
        try:
            parsed_url = urlparse(self.prefix)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"

            self._robot_parser = RobotFileParser()
            self._robot_parser.set_url(robots_url)
            self._robot_parser.read()
            logger.info(f"Successfully loaded robots.txt from {robots_url}")
        except Exception as e:
            logger.warning(
                f"Failed to load robots.txt: {e}. Proceeding without robots.txt restrictions."
            )
            self._robot_parser = None

    def _can_fetch(self, url: str) -> bool:
        """
        Check if the given URL can be fetched according to robots.txt rules.

        Args:
            url (str): The URL to check.

        Returns:
            bool: True if the URL can be fetched, False otherwise.

        """
        if not self.respect_robots_txt or self._robot_parser is None:
            return True

        try:
            can_fetch = self._robot_parser.can_fetch("*", url)
            if not can_fetch:
                logger.info(f"URL disallowed by robots.txt: {url}")
            return can_fetch
        except Exception as e:
            logger.warning(f"Error checking robots.txt for {url}: {e}. Allowing fetch.")
            return True

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

    def load_data(
        self,
        base_url: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Document]:
        """
        Load data from the base URL using BFS algorithm.

        Args:
            base_url (str): Base URL to start scraping.
            progress_callback (Optional[Callable[[Dict[str, Any]], None]]): Optional callback
                function to track progress. The callback receives a dictionary with keys:
                'current_url', 'depth', 'pages_visited', 'pages_remaining', 'total_pages_found'.

        Returns:
            List[Document]: List of scraped documents.

        """
        added_urls = set()
        urls_to_visit = [(base_url, 0)]
        documents = []
        pages_visited = 0

        while urls_to_visit:
            current_url, depth = urls_to_visit.pop(0)
            logger.info(
                f"Visiting: {current_url}, {len(urls_to_visit)} pages remaining"
            )

            # Call progress callback if provided
            if progress_callback:
                progress_callback(
                    {
                        "current_url": current_url,
                        "depth": depth,
                        "pages_visited": pages_visited,
                        "pages_remaining": len(urls_to_visit),
                        "total_pages_found": len(added_urls),
                    }
                )

            # Check robots.txt before visiting
            if not self._can_fetch(current_url):
                logger.info(f"Skipping URL disallowed by robots.txt: {current_url}")
                continue

            try:
                self.driver.get(current_url)
                page_content = self.extract_content()
                added_urls.add(current_url)
                pages_visited += 1

                next_depth = depth + 1
                if next_depth <= self.max_depth:
                    # links = self.driver.find_elements(By.TAG_NAME, 'a')
                    links = self.extract_links()
                    # clean all urls
                    links = [self.clean_url(link) for link in links]
                    # extract new links
                    links = [link for link in links if link not in added_urls]
                    logger.info(
                        f"Found {len(links)} new potential links at depth {depth}"
                    )

                    for href in links:
                        try:
                            if href.startswith(self.prefix) and href not in added_urls:
                                urls_to_visit.append((href, next_depth))
                                added_urls.add(href)
                        except Exception as e:
                            logger.debug(f"Error processing link {href}: {e}")
                            continue

                doc = Document(text=page_content, extra_info={"URL": current_url})
                if self.uri_as_id:
                    warnings.warn(
                        "Setting the URI as the id of the document might break the code execution downstream and should be avoided."
                    )
                    doc.id_ = current_url
                documents.append(doc)
                logger.debug(f"Successfully scraped {current_url}")
                time.sleep(self.delay)

            except WebDriverException as e:
                logger.error(
                    f"WebDriverException encountered: {e}. Restarting driver..."
                )
                self.restart_driver()
            except Exception as e:
                logger.error(
                    f"An unexpected exception occurred: {e}. Skipping URL: {current_url}"
                )
                continue

        self.driver.quit()
        logger.info(
            f"Scraping complete. Visited {pages_visited} pages, collected {len(documents)} documents."
        )
        return documents
