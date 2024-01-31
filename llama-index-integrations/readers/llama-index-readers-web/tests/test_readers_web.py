from llama_index.core.readers.base import BaseReader
from llama_index.readers.web import (
    AsyncWebPageReader,
    BeautifulSoupWebReader,
    KnowledgeBaseWebReader,
    MainContentExtractorReader,
    NewsArticleReader,
    RssNewsReader,
    ReadabilityWebPageReader,
    RssReader,
    SimpleWebPageReader,
    SitemapReader,
    TrafilaturaWebReader,
    UnstructuredURLLoader,
    WholeSiteReader,
)


def test_class():
    names_of_base_classes = [b.__name__ for b in AsyncWebPageReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in BeautifulSoupWebReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in KnowledgeBaseWebReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in MainContentExtractorReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in NewsArticleReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in RssNewsReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in ReadabilityWebPageReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in RssReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in SimpleWebPageReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in SitemapReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in TrafilaturaWebReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in UnstructuredURLLoader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in WholeSiteReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes
