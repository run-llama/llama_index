# from llama_index.core import SummaryIndex
# from llama_index.core.readers.base import BaseReader
# from llama_index.readers.web.spider_web.base import SpiderWebReader

# import os

# os.environ["OPENAI_API_KEY"] = "sk-proj-bKek612HDgEO7F4jsRd7T3BlbkFJoZecgNPclOalvDfSu4kl"


# def test_class():
#     names_of_base_classes = [b.__name__ for b in SpiderWebReader.__mro__]
#     assert BaseReader.__name__ in names_of_base_classes

# def test_spider_base_functionality():
#     reader = SpiderWebReader(api_key="sk-dba71dae-8ed2-4993-a0ba-596d2d763572", mode="scrape")
#     result = reader.load_data("https://spider.cloud")
#     assert result is not None
#     print("Document returned:", result)
#     index = SummaryIndex.from_documents(result)
#     query_engine = index.as_query_engine()
#     response = query_engine.query("What is spider")
#     print(response)
#     # assert result == isinstance(list)
# test_spider_base_functionality()

from llama_index.core import SummaryIndex
from llama_index.readers.web import SpiderWebReader
import os

os.environ[
    "OPENAI_API_KEY"
] = "sk-proj-bKek612HDgEO7F4jsRd7T3BlbkFJoZecgNPclOalvDfSu4kl"


firecrawl_reader = SpiderWebReader(
    api_key="sk-dba71dae-8ed2-4993-a0ba-596d2d763572",  # Replace with your actual API key from https://www.firecrawl.dev/
    mode="crawl",  # Choose between "crawl" and "scrape" for single page scraping
)

documents = firecrawl_reader.load_data(url="https://spider.cloud")

print(documents)

index = SummaryIndex.from_documents(documents)

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("Describe spider in this context")
print(response)
