"""Init file."""
from llama_index.readers.web.async_web.base import (
    AsyncWebPageReader,
)
from llama_index.readers.web.beautiful_soup_web.base import (
    BeautifulSoupWebReader,
)
from llama_index.readers.web.knowledge_base.base import (
    KnowledgeBaseWebReader,
)
from llama_index.readers.web.main_content_extractor.base import (
    MainContentExtractorReader,
)
from llama_index.readers.web.news.base import NewsArticleReader
from llama_index.readers.web.readability_web.base import (
    ReadabilityWebPageReader,
)
from llama_index.readers.web.rss.base import (
    RssReader,
)
from llama_index.readers.web.rss_news.base import (
    RssNewsReader,
)
from llama_index.readers.web.simple_web.base import (
    SimpleWebPageReader,
)
from llama_index.readers.web.sitemap.base import (
    SitemapReader,
)
from llama_index.readers.web.trafilatura_web.base import (
    TrafilaturaWebReader,
)
from llama_index.readers.web.unstructured_web.base import (
    UnstructuredURLLoader,
)
from llama_index.readers.web.whole_site.base import (
    WholeSiteReader,
)

__all__ = [
    "AsyncWebPageReader",
    "BeautifulSoupWebReader",
    "KnowledgeBaseWebReader",
    "MainContentExtractorReader",
    "NewsArticleReader",
    "ReadabilityWebPageReader",
    "RssReader",
    "RssNewsReader",
    "SimpleWebPageReader",
    "SitemapReader",
    "TrafilaturaWebReader",
    "UnstructuredURLLoader",
    "WholeSiteReader",
]
