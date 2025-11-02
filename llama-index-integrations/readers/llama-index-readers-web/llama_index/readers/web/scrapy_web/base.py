import json
import os
from typing import List, Optional, Dict, Union

from scrapy.spiders import Spider, signals
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document


class ScrapyWebReader(BasePydanticReader):
    """
    Scrapy web page reader.

    Reads pages from the web.

    Args:
        project_path (Optional[str]): The path to the Scrapy project for
            loading the project settings (with middlewares and pipelines).
            The project path should contain the `scrapy.cfg` file.
            Settings will be set to empty if path not specified or not found.
            Defaults to "".

        metadata_keys (Optional[List[str]]): List of keys to use
            as document metadata from the scraped item. Defaults to [].

        keep_keys (bool): Whether to keep metadata keys in items.
            Defaults to False.
    """

    project_path: Optional[str] = ""
    metadata_keys: Optional[List[str]] = []
    keep_keys: bool = False
    _settings: Dict = PrivateAttr()

    def __init__(self, project_path: Optional[str] = "",
                 metadata_keys: Optional[List[str]] = [],
                 keep_keys: bool = False):
        super().__init__(
            project_path=project_path,
            metadata_keys=metadata_keys,
            keep_keys=keep_keys,
        )

        self._settings = self._load_settings()

    @classmethod
    def class_name(cls) -> str:
        return "ScrapyWebReader"

    def load_data(self, spider: Union[Spider, str]) -> List[Document]:
        """
        Load data from the input spider.

        Args:
            spider (Union[Spider, str]): The Scrapy spider class or
                the spider name from the project to use for scraping.

        Returns:
            List[Document]: List of documents extracted from the web pages.
        """

        documents = []

        def item_scraped(item, response, spider):
            documents.append(self._item_to_document(dict(item)))

        process = CrawlerProcess(settings=self._settings)
        crawler = process.create_crawler(spider)
        crawler.signals.connect(item_scraped, signal=signals.item_scraped)

        process.crawl(crawler)
        process.start()

        return documents

    def _item_to_document(self, item: Dict) -> Document:
        metadata = self._setup_metadata(item)
        item = self._remove_metadata_keys(item)

        return Document(text=json.dumps(item), metadata=metadata)

    def _setup_metadata(self, item: Dict) -> Dict:
        metadata = {}

        for key in self.metadata_keys:
            if key in item:
                metadata[key] = item[key]

        return metadata

    def _remove_metadata_keys(self, item: Dict) -> Dict:
        if not self.keep_keys:
            for key in self.metadata_keys:
                item.pop(key, None)

        return item

    def _load_settings(self) -> Dict:
        if not self.project_path:
            return {}

        if not os.path.exists(self.project_path):
            return {}

        os.chdir(self.project_path)

        return get_project_settings() or {}
