import os
from pathlib import Path
from llama_index import GPTSimpleKeywordTableIndex, LLMPredictor, SimpleDirectoryReader
import requests
from dotenv import load_dotenv
import logging
import sys
from llama_index import GPTVectorStoreIndex, ServiceContext, StorageContext
from langchain.llms.openai import OpenAIChat
from langchain.chat_models import ChatOpenAI
from llama_index.indices.composability import ComposableGraph
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform

# Load environment variables from the .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

wiki_titles = ["player-characters", "expeditions"]
documents = {}
for title in wiki_titles:
    documents[title] = SimpleDirectoryReader(input_dir=f"data/{title}").load_data()
    # TODO: Is the input_dir the problem, should be input_files? LOOKS LIKE IT'S NOT INDEXING PROPERLY.

######## Defining the Set of Indexes

llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, chunk_size_limit=1024
)
print("service_context.chunk_size_limit: ", service_context.chunk_size_limit)


vector_indexes = {}
for wiki_title in wiki_titles:
    storage_context = StorageContext.from_defaults()
    vector_indexes[wiki_title] = GPTVectorStoreIndex.from_documents(
        documents[wiki_title],
        service_context=service_context,
        storage_context=storage_context,
    )
    vector_indexes[wiki_title].index_struct.index_id = wiki_title

# print(vector_indexes)

# This will call the API.
# query_engine = vector_indexes["characters"].as_query_engine()
# response = query_engine.query("who is timou?")
# print(str(response))

############### Defining a Graph for Compare/Contrast Queries
index_summaries = {}
for wiki_title in wiki_titles:
    # set summary for text file.
    index_summaries[wiki_title] = (
        f"This content contains articles about {wiki_title}."
        f"Use this index if you need to lookup specific facts about {wiki_title}."
    )

graph = ComposableGraph.from_indices(
    GPTSimpleKeywordTableIndex,
    [index for _, index in vector_indexes.items()],
    [summary for _, summary in index_summaries.items()],
    max_keywords_per_chunk=50,
)

# get root index
# documentation is all wrong here.
root_index = graph.get_index(graph.index_struct.index_id)
root_index.set_index_id("compare_contrast")
root_summary = (
    "This index contains articles about the fictional Dungeons and Dragons world Kazar."
    "Use this index if you want to compare things about the world."
)


# define decompose_transform
decompose_transform = DecomposeQueryTransform(llm_predictor, verbose=True)

# define custom query engines
from llama_index.query_engine.transform_query_engine import TransformQueryEngine

custom_query_engines = {}
for index in vector_indexes.values():
    query_engine = index.as_query_engine(service_context=service_context)
    query_engine = TransformQueryEngine(
        query_engine,
        query_transform=decompose_transform,
        transform_extra_info={"index_summary": index.index_struct.summary},
    )
    custom_query_engines[index.index_id] = query_engine
custom_query_engines[graph.root_id] = graph.root_index.as_query_engine(
    retriever_mode="simple",
    response_mode="tree_summarize",
    service_context=service_context,
)

# define query engine
query_engine = graph.as_query_engine(custom_query_engines=custom_query_engines)
print("QUERY ENGINE")
print(query_engine)

# query the graph
response = query_engine.query("how are you?")

# https://chat.openai.com/c/f5742f35-fce9-4659-b993-5b14de9e383f

# <llama_index.query_engine.graph_query_engine.ComposableGraphQueryEngine object at 0x7fd4d0e61d80>
# INFO:llama_index.indices.keyword_table.retrievers:> Starting query: how are you?
# INFO:llama_index.indices.keyword_table.retrievers:query keywords: []
# INFO:llama_index.indices.keyword_table.retrievers:> Extracted keywords: []
# Warning: num_chunks text splitter was zero, setting to 1 to avoid division by zero
# INFO:llama_index.token_counter.token_counter:> [get_response] Total LLM token usage: 0 tokens
# INFO:llama_index.token_counter.token_counter:> [get_response] Total embedding token usage: 0 tokens
# INFO:llama_index.token_counter.token_counter:> [get_response] Total LLM token usage: 0 tokens
# INFO:llama_index.token_counter.token_counter:> [get_response] Total embedding token usage: 0 tokens
