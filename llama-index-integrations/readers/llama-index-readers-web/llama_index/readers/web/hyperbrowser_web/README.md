# Hyperbrowser Web Loader

[Hyperbrowser](https://hyperbrowser.ai) is a platform for running and scaling headless browsers. It lets you launch and manage browser sessions at scale and provides easy to use solutions for any webscraping needs, such as scraping a single page or crawling an entire site.

Key Features:

- Instant Scalability - Spin up hundreds of browser sessions in seconds without infrastructure headaches
- Simple Integration - Works seamlessly with popular tools like Puppeteer and Playwright
- Powerful APIs - Easy to use APIs for scraping/crawling any site, and much more
- Bypass Anti-Bot Measures - Built-in stealth mode, ad blocking, automatic CAPTCHA solving, and rotating proxies

## Setup and Installation

- Head to [Hyperbrowser](https://app.hyperbrowser.ai/) to sign up and generate an API key. Once you've done this set the HYPERBROWSER_API_KEY environment variable or you can pass it to the `HyperbrowserWebReader` constructor.
- Install the [Hyperbrowser SDK](https://github.com/hyperbrowserai/python-sdk):

```bash
pip install hyperbrowser
```

## Usage

Once you have your API Key and have installed the SDK you can load webpages into LlamaIndex using `HyperbrowserWebReader`.

```python
from llama_index.readers.web import HyperbrowserWebReader

reader = HyperbrowserWebReader(api_key="your_api_key_here")
```

To load data, you can specify the operation to be performed by the loader. The default operation is `scrape`. For `scrape`, you can provide a single URL or a list of URLs to be scraped. For `crawl`, you can only provide a single URL. The `crawl` operation will crawl the provided page and subpages and return a document for each page. HyperbrowserWebReader supports loading and lazy loading data in both sync and async modes.

```python
documents = reader.load_data(
    urls=["https://example.com"],
    operation="scrape",
)
```

Optional params for the loader can also be provided in the `params` argument. For more information on the supported params, visit https://docs.hyperbrowser.ai/reference/sdks/python/scrape#start-scrape-job-and-wait or https://docs.hyperbrowser.ai/reference/sdks/python/crawl#start-crawl-job-and-wait.

```python
documents = reader.load_data(
    urls=["https://example.com"],
    operation="scrape",
    params={"scrape_options": {"include_tags": ["h1", "h2", "p"]}},
)
```

## Additional Links

- [Hyperbrowser](https://hyperbrowser.ai)
- [Hyperbrowser Documentation](https://docs.hyperbrowser.ai/)
- [Hyperbrowser SDK](https://github.com/hyperbrowserai/python-sdk)
