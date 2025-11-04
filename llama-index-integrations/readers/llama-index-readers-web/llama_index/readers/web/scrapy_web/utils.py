import json
import os
from multiprocessing import Queue
from typing import Dict

from scrapy.spiders import signals, Spider
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from llama_index.core.schema import Document


def load_scrapy_settings(project_path: str) -> Dict:
    """
    Load Scrapy settings from the given project path.
    """
    if not project_path:
        return {}

    if not os.path.exists(project_path):
        return {}

    cwd = os.getcwd()

    try:
        os.chdir(project_path)

        try:
            settings = get_project_settings() or {}
        except Exception:
            settings = {}
    finally:
        os.chdir(cwd)

    return settings


def run_spider_process(spider: Spider, documents_queue: Queue, config: Dict):
    """
    Run the Scrapy spider process and collect documents in the queue.
    """
    documents = []

    def item_scraped(item, response, spider):
        documents.append(item_to_document(dict(item), config))

    process = CrawlerProcess(settings=config["settings"])
    crawler = process.create_crawler(spider)
    crawler.signals.connect(item_scraped, signal=signals.item_scraped)
    process.crawl(crawler)
    process.start()

    documents_queue.put(documents)


def item_to_document(item: Dict, config: Dict) -> Dict:
    """
    Convert a scraped item to a Document with metadata.
    """
    metadata = setup_metadata(item, config)
    item = remove_metadata_keys(item, config)

    return Document(text=json.dumps(item), metadata=metadata)


def setup_metadata(item: Dict, config: Dict) -> Dict:
    """
    Set up metadata for the document from the scraped item.
    """
    metadata = {}

    for key in config["metadata_keys"]:
        if key in item:
            metadata[key] = item[key]

    return metadata


def remove_metadata_keys(item: Dict, config: Dict) -> Dict:
    """
    Remove metadata keys from the scraped item if keep_keys is False.
    """
    if not config["keep_keys"]:
        for key in config["metadata_keys"]:
            item.pop(key, None)

    return item
