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

######## Defining the Set of Indexes

llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, chunk_size_limit=1024
)

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

index_summaries = {}
for wiki_title in wiki_titles:
    # set summary for text file.
    index_summaries[wiki_title] = (
        f"This content contains articles about {wiki_title}. "
        f"Use this index if you need to lookup specific facts about {wiki_title}. "
    )

# print(index_summaries)

graph = ComposableGraph.from_indices(
    GPTSimpleKeywordTableIndex,
    [index for _, index in vector_indexes.items()],
    [summary for _, summary in index_summaries.items()],
    max_keywords_per_chunk=50,
)

# get root index
root_index = graph.get_index(graph.index_struct.root_id, GPTSimpleKeywordTableIndex)


# This will call the API.
# query_engine = vector_indexes["characters"].as_query_engine()
# response = query_engine.query("who is timou?")
# print(str(response))

############### Defining a Graph for Compare/Contrast Queries
