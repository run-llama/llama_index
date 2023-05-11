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
from llama_index.query_engine.transform_query_engine import TransformQueryEngine


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

# This will call the API.
# query_engine = vector_indexes["player-characters"].as_query_engine()
# response = query_engine.query("who is timou?")
# print(str(response))

############## Defining a Graph for Compare/Contrast Queries
index_summaries = {}
for wiki_title in wiki_titles:
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
root_index = graph.get_index(graph.index_struct.index_id)
root_index.set_index_id("compare_contrast")
root_summary = (
    "This index contains articles about the fictional Dungeons and Dragons world Kazar."
    "Use this index if you want to compare things about the world."
)


# define decompose_transform
decompose_transform = DecomposeQueryTransform(llm_predictor, verbose=True)

# define custom query engines
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
query_engine = graph.as_query_engine(custom_query_engines)

# query the graph
response = query_engine.query("Compare Timou to Thaddeus")

# https://chat.openai.com/c/f5742f35-fce9-4659-b993-5b14de9e383f
