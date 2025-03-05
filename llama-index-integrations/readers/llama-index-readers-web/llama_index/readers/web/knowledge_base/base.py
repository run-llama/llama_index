from typing import Any, Dict, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class KnowledgeBaseWebReader(BaseReader):
    """
    Knowledge base reader.

    Crawls and reads articles from a knowledge base/help center with Playwright.
    Tested on Zendesk and Intercom CMS, may work on others.
    Can be run in headless mode but it may be blocked by Cloudflare. Run it headed to be safe.
    Times out occasionally, just increase the default time out if it does.
    Requires the `playwright` package.

    Args:
        root_url (str): the base url of the knowledge base, with no trailing slash
            e.g. 'https://support.intercom.com'
        link_selectors (List[str]): list of css selectors to find links to articles while crawling
            e.g. ['.article-list a', '.article-list a']
        article_path (str): the url path of articles on this domain so the crawler knows when to stop
            e.g. '/articles'
        title_selector (Optional[str]): css selector to find the title of the article
            e.g. '.article-title'
        subtitle_selector (Optional[str]): css selector to find the subtitle/description of the article
            e.g. '.article-subtitle'
        body_selector (Optional[str]): css selector to find the body of the article
            e.g. '.article-body'
    """

    def __init__(
        self,
        root_url: str,
        link_selectors: List[str],
        article_path: str,
        title_selector: Optional[str] = None,
        subtitle_selector: Optional[str] = None,
        body_selector: Optional[str] = None,
        max_depth: int = 100,
    ) -> None:
        """Initialize with parameters."""
        self.root_url = root_url
        self.link_selectors = link_selectors
        self.article_path = article_path
        self.title_selector = title_selector
        self.subtitle_selector = subtitle_selector
        self.body_selector = body_selector
        self.max_depth = max_depth

    def load_data(self) -> List[Document]:
        """Load data from the knowledge base."""
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)

            # Crawl
            article_urls = self.get_article_urls(
                browser, self.root_url, self.root_url, self.max_depth
            )

            # Scrape
            documents = []
            for url in article_urls:
                article = self.scrape_article(
                    browser,
                    url,
                )
                extra_info = {
                    "title": article["title"],
                    "subtitle": article["subtitle"],
                    "url": article["url"],
                }
                documents.append(Document(text=article["body"], extra_info=extra_info))

            browser.close()

            return documents

    def scrape_article(
        self,
        browser: Any,
        url: str,
    ) -> Dict[str, str]:
        """
        Scrape a single article url.

        Args:
            browser (Any): a Playwright Chromium browser.
            url (str): URL of the article to scrape.

        Returns:
            Dict[str, str]: a mapping of article attributes to their values.

        """
        page = browser.new_page(ignore_https_errors=True)
        page.set_default_timeout(60000)
        page.goto(url, wait_until="domcontentloaded")

        title = (
            (
                page.query_selector(self.title_selector).evaluate(
                    "node => node.innerText"
                )
            )
            if self.title_selector
            else ""
        )
        subtitle = (
            (
                page.query_selector(self.subtitle_selector).evaluate(
                    "node => node.innerText"
                )
            )
            if self.subtitle_selector
            else ""
        )
        body = (
            (page.query_selector(self.body_selector).evaluate("node => node.innerText"))
            if self.body_selector
            else ""
        )

        page.close()
        print("scraped:", url)
        return {"title": title, "subtitle": subtitle, "body": body, "url": url}

    def get_article_urls(
        self,
        browser: Any,
        root_url: str,
        current_url: str,
        max_depth: int = 100,
        depth: int = 0,
    ) -> List[str]:
        """
        Recursively crawl through the knowledge base to find a list of articles.

        Args:
            browser (Any): a Playwright Chromium browser.
            root_url (str): root URL of the knowledge base.
            current_url (str): current URL that is being crawled.
            max_depth (int): maximum recursion level for the crawler
            depth (int): current depth level

        Returns:
            List[str]: a list of URLs of found articles.

        """
        if depth >= max_depth:
            print(f"Reached max depth ({max_depth}): {current_url}")
            return []

        page = browser.new_page(ignore_https_errors=True)
        page.set_default_timeout(60000)
        page.goto(current_url, wait_until="domcontentloaded")

        # If this is a leaf node aka article page, return itself
        if self.article_path in current_url:
            print("Found an article: ", current_url)
            page.close()
            return [current_url]

        # Otherwise crawl this page and find all the articles linked from it
        article_urls = []
        links = []

        for link_selector in self.link_selectors:
            ahrefs = page.query_selector_all(link_selector)
            links.extend(ahrefs)

        for link in links:
            url = root_url + page.evaluate("(node) => node.getAttribute('href')", link)
            article_urls.extend(
                self.get_article_urls(browser, root_url, url, max_depth, depth + 1)
            )

        page.close()

        return article_urls
