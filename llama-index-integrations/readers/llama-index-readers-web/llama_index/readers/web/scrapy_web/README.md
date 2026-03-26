# LlamaIndex Scrapy Web Reader Integration

This integration provides the `ScrapyWebReader` class that allows you to use Scrapy to scrape data and load it into LlamaIndex.

## Installation

```bash
pip install llama-index llama-index-readers-web
```

## Usage

The `ScrapyWebReader` can be used in 2 ways

1. By providing an Scrapy spider class.
2. By providing the path to a Scrapy project.

### 1. Using with Scrapy Spider Class

```python
from llama_index.readers.web import ScrapyWebReader


class SampleSpider(Spider):
    name = "sample_spider"
    start_urls = ["http://quotes.toscrape.com"]

    def parse(self, response):
        ...


reader = ScrapyWebReader()
docs = reader.load_data(SampleSpider)
```

### 2. Using with Scrapy Project Path

```python
from llama_index.readers.web import ScrapyWebReader

reader = ScrapyWebReader(project_path="/path/to/scrapy/project")
docs = reader.load_data("spider_name")
```
