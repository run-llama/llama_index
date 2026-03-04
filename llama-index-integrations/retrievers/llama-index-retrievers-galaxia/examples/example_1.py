#! /usr/bin/env python3

from llama_index.core.schema import QueryBundle
from llama_index.retrievers.galaxia import GalaxiaRetriever

retriever = GalaxiaRetriever(
    api_url="https://beta.api.smabbler.com",
    api_key="<key>",
    knowledge_base_id="<knowledge_base_id>",
)

query = "What is Marie Curie's nationality?"

result = gr.retrieve(QueryBundle(query))

print(result)
