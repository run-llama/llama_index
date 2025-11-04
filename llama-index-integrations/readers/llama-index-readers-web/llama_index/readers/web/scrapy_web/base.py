from typing import List, Optional, Union
from multiprocessing import Process, Queue

from scrapy.spiders import Spider

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

from .utils import run_spider_process, load_scrapy_settings


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

    def __init__(
        self,
        project_path: Optional[str] = "",
        metadata_keys: Optional[List[str]] = [],
        keep_keys: bool = False,
    ):
        super().__init__(
            project_path=project_path,
            metadata_keys=metadata_keys,
            keep_keys=keep_keys,
        )

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
        if not self._is_spider_correct_type(spider):
            raise ValueError(
                "Invalid spider type. Provide a Spider class or spider name with project path."
            )

        documents_queue = Queue()

        config = {
            "keep_keys": self.keep_keys,
            "metadata_keys": self.metadata_keys,
            "settings": load_scrapy_settings(self.project_path),
        }

        # Running each spider in a separate process as Scrapy uses
        # twisted reactor which can only be run once in a process
        process = Process(
            target=run_spider_process, args=(spider, documents_queue, config)
        )

        process.start()
        process.join()

        if documents_queue.empty():
            return []

        return documents_queue.get()

    def _is_spider_correct_type(self, spider: Union[Spider, str]) -> bool:
        return not (isinstance(spider, str) and not self.project_path)
