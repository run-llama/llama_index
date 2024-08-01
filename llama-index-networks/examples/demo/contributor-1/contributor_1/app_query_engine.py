"""Contributor Service #1.

This builds a RAG over a Paul Graham Essay: https://paulgraham.com/articles.html
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


# create the pipeline with transformations
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
    ]
)

# build the index
loader = SimpleDirectoryReader(input_dir="./data")
documents = loader.load_data()
nodes = pipeline.run(documents=documents, show_progress=True)

# models
llm = OpenAI()
embed_model = OpenAIEmbedding()

# build RAG
index = VectorStoreIndex(nodes=nodes, embed_model=embed_model)
query_engine = index.as_query_engine(llm=llm)
