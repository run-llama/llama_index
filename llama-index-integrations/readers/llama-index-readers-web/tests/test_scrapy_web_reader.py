import io
import pytest
import zipfile

import requests

from llama_index.readers.web import ScrapyWebReader

try:
    from scrapy.spiders import Spider

    SCRAPY_AVAILABLE = True
except ImportError:
    SCRAPY_AVAILABLE = False

pytestmark = pytest.mark.skipif(not SCRAPY_AVAILABLE, reason="Scrapy not installed")


class SampleSpider(Spider):
    name = "sample_spider"
    start_urls = ["http://quotes.toscrape.com"]

    def parse(self, response):
        for q in response.css("div.quote"):
            yield {
                "text": q.css("span.text::text").get(),
                "author": q.css(".author::text").get(),
            }


@pytestmark
def test_scrapy_web_reader_with_spider_class():
    reader = ScrapyWebReader()
    docs = reader.load_data(SampleSpider)

    assert isinstance(docs, list)
    assert len(docs) > 0


@pytestmark
def test_scrapy_web_reader_with_zip_project(tmp_path):
    project_zip_url = (
        "https://github.com/scrapy/quotesbot/archive/refs/heads/master.zip"
    )
    response = requests.get(project_zip_url)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extractall(tmp_path)

    project_dir = tmp_path / "quotesbot-master"
    reader = ScrapyWebReader(project_path=str(project_dir))
    docs = reader.load_data("toscrape-css")

    assert isinstance(docs, list)
    assert len(docs) > 0
