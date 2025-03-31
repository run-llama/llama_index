#! /usr/bin/env python3

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.retrievers.galaxia import GalaxiaRetriever

api_key = "<your API key>"
api_url = "https://dev.api.smabbler.com"
knowledge_base_id = "<your KB ID>"

gr = GalaxiaRetriever(
    api_url,
    api_key,
    knowledge_base_id,
)

query = "What is Marie Curie's nationality?"

result = gr.retrieve(QueryBundle(query))

print(result)
