"""
Runs Graph RAG with a SPARQL server as storage

Preparation :

* pip install openai
* pip install sparqlwrapper
* make a SPARQL endpoint available, add URL below (make sure it supports UPDATE, as /llama_index_sparql-test/)
* for a clean start DROP GRAPH <http://purl.org/stuff/guardians>
* add OpenAI API key below

@danja 2023-09-17
"""

# import llama_index 
from llama_index.readers.download import download_loader
# from llama_index import download_loader
import os
import logging
from llama_index import (
    KnowledgeGraphIndex,
    ServiceContext,
)

from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import SparqlGraphStore
from llama_index.llms import OpenAI
from IPython.display import Markdown, display
from llama_index import load_index_from_storage
import os
import openai

logging.basicConfig(filename='loggy.log', filemode='w', level=logging.DEBUG)
logger = logging.getLogger(__name__)

############
# LLM Config
############
os.environ["OPENAI_API_KEY"] = ""

openai.api_key = ""

llm = OpenAI(temperature=0, model="text-davinci-002")
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)

###############
# SPARQL Config
###############
ENDPOINT = 'https://fuseki.hyperdata.it/llama_index_sparql-test/'
GRAPH = 'http://purl.org/stuff/guardians'
BASE_URI = 'http://purl.org/stuff/data'

graph_store = SparqlGraphStore(
    sparql_endpoint=ENDPOINT,
    sparql_graph=GRAPH,
    sparql_base_uri=BASE_URI,
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)


WikipediaReader = download_loader("WikipediaReader")
loader = WikipediaReader()
documents = loader.load_data(
    pages=['Guardians of the Galaxy Vol. 3'], auto_suggest=False)

kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    service_context=service_context,
    max_triplets_per_chunk=10,
    sparql_endpoint=ENDPOINT,
    sparql_graph=GRAPH,
    sparql_base_uri=BASE_URI,
    include_embeddings=True,
)


# print('*** Persist to/Load from local disk ***')

"""
storage_context = StorageContext.from_defaults(
    persist_dir='./storage_graph', graph_store=graph_store)
kg_index = load_index_from_storage(
    storage_context=storage_context,
    service_context=service_context,
    include_embeddings=True,
    sparql_endpoint=ENDPOINT,  # shouldn't be needed
    sparql_graph=GRAPH,
    sparql_base_uri=BASE_URI,
)
"""

# FileNotFoundError: [Errno 2] No such file or directory: '/home/danny/AI/nlp/GraphRAG/src/storage_graph/docstore.json'
# copied files I found in a storage_vector/docstore.json into /home/danny/AI/nlp/GraphRAG/src/storage_graph/

# print('*** Prepare Graph RAG query engine***')
kg_rag_query_engine = kg_index.as_query_engine(
    include_text=False,
    retriever_mode="keyword",
    # RecursionError: maximum recursion depth exceeded in comparison
    response_mode="tree_summarize",
)

# print('*** Do query ***')
# response_graph_rag = kg_rag_query_engine.query(
#    "What do cats eat?")
# print(str(response_graph_rag))

response_graph_rag = kg_rag_query_engine.query(
    "Who is Quill?")
print(str(response_graph_rag))

# display(Markdown(f"<b>{response_graph_rag}</b>"))
