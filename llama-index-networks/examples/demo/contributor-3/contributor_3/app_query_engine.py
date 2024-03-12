"""Contributor Service #3.

This service exposes a query engine built over Wikipedia pages of various cities.
"""

from llama_index.core import VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.readers.wikipedia import WikipediaReader


# create the pipeline with transformations
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
    ]
)

# build the index
# wikipedia pages
cities = [
    "San Francisco",
    "Toronto",
    "New York City",
    "Vancouver",
    "Montreal",
    "Tokyo",
    "Singapore",
    "Paris",
]

documents = WikipediaReader().load_data(pages=[f"History of {x}" for x in cities])
nodes = pipeline.run(documents=documents, show_progress=True)

# models
llm = OpenAI()
embed_model = OpenAIEmbedding()

# build RAG
index = VectorStoreIndex(nodes=nodes, embed_model=embed_model)
query_engine = index.as_query_engine(llm=llm)
