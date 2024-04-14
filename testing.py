from llama_index.core import SummaryIndex
from llama_index.readers.web import SimpleWebPageReader
from IPython.display import Markdown, display
import os
from IPython.display import Markdown, display
import os
from llama_index.readers.web import FireCrawlWebReader


print("hello world")


documents = FireCrawlWebReader(
    api_key="30c90634-8377-4446-9ef9-a280b9be1702", # get one here -> https://www.firecrawl.dev/
    mode="crawl"
).load_data(
  url="http://paulgraham.com/"
)
   

print("hello world 2")

print(len(documents))


index = SummaryIndex.from_documents(documents)

print("hello world 3")

query_engine = index.as_query_engine()

print("hello world 4")

response = query_engine.query("What did Paul Graham do growing up?")

display(Markdown(f"<b>{response}</b>"))
print(response)

# documents = SimpleWebPageReader(html_to_text=True).load_data(
#     ["http://paulgraham.com/worked.html"]
# )
# print('documents', documents[0])
# index = SummaryIndex.from_documents(documents)



# # set Logging to DEBUG for more detailed outputs
# query_engine = index.as_query_engine()
# response = query_engine.query("What did the author do growing up?")


# display(Markdown(f"<b>{response}</b>"))