import os

import openai
from llama_index import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.readers.semanticscholar.base import SemanticScholarReader

# initialize the SemanticScholarReader
s2reader = SemanticScholarReader()

# initialize the service context
openai.api_key = os.environ["OPENAI_API_KEY"]

query_space = "large language models"
query_string = "limitations of using large language models"
full_text = True
# be careful with the total_papers when full_text = True
# it can take a long time to download
total_papers = 50

persist_dir = (
    "./citation_" + query_space + "_" + str(total_papers) + "_" + str(full_text)
)


if not os.path.exists(persist_dir):
    # Load data from Semantic Scholar
    documents = s2reader.load_data(query_space, total_papers, full_text=full_text)
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=persist_dir)
else:
    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=persist_dir),
    )
# initialize the citation query engine
query_engine = CitationQueryEngine.from_args(
    index,
    similarity_top_k=3,
    citation_chunk_size=512,
)

# query the citation query engine
response = query_engine.query(query_string)
print("Answer: ", response)
print("Source nodes: ")
for node in response.source_nodes:
    print(node.node.metadata)

"""
output = (
    "Output:\n"
    "Answer:  The limitations of using large language models include the struggle "
    "to learn long-tail knowledge [2],\n"
    "the need for scaling by many orders of magnitude to reach competitive "
    "performance on questions with little support in the pre-training data [2],\n"
    "and the difficulty in synthesizing complex programs from natural language "
    "descriptions [3].\n"
    "Source nodes:\n"
    "{'venue': 'arXiv.org', 'year': 2022, 'paperId': '3eed4de25636ac90f39f6e1ef70e3507ed61a2a6', "
    "'citationCount': 35, 'openAccessPdf': None, 'authors': ['M. Shanahan'], "
    "'title': 'Talking About Large Language Models'}\n"
    "{'venue': 'arXiv.org', 'year': 2022, 'paperId': '6491980820d9c255b9d798874c8fce696750e0d9', "
    "'citationCount': 31, 'openAccessPdf': None, 'authors': ['Nikhil Kandpal', 'H. Deng', "
    "'Adam Roberts', 'Eric Wallace', 'Colin Raffel'], "
    "'title': 'Large Language Models Struggle to Learn Long-Tail Knowledge'}\n"
    "{'venue': 'arXiv.org', 'year': 2021, 'paperId': 'a38e0f993e4805ba8a9beae4c275c91ffcec01df', "
    "'citationCount': 305, 'openAccessPdf': None, 'authors': ['Jacob Austin', 'Augustus Odena', "
    "'Maxwell Nye', 'Maarten Bosma', 'H. Michalewski', 'David Dohan', 'Ellen Jiang', 'Carrie J. Cai', "
    "'Michael Terry', 'Quoc V. Le', 'Charles Sutton'], 'title': 'Program Synthesis with Large Language Models'}"
)
"""
