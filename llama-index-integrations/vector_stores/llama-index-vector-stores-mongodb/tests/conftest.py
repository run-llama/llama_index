import os
from typing import List

import pytest
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.llms.azure_openai import AzureOpenAI
from pymongo import MongoClient

import threading

lock = threading.Lock()


@pytest.fixture(scope="session")
def embed_model() -> OpenAIEmbedding:
    if "OPENAI_API_KEY" in os.environ:
        return OpenAIEmbedding()
    if "AZURE_OPENAI_API_KEY" in os.environ:
        deployment_name = os.environ.get(
            "AZURE_TEXT_DEPLOYMENT", "text-embedding-3-small"
        )
        api_key = os.environ["AZURE_OPENAI_API_KEY"]
        embedding = AzureOpenAIEmbedding(
            api_key=api_key, deployment_name=deployment_name
        )
        Settings.embed_model = embedding
        return embedding
    pytest.skip("Requires OPENAI_API_KEY or AZURE_OPENAI_API_KEY in os.environ")


@pytest.fixture(scope="session")
def documents() -> List[Document]:
    """
    List of documents represents data to be embedded in the datastore.
    Minimum requirements for Documents in the /upsert endpoint's UpsertRequest.
    """
    text = Document.example().text
    metadata = Document.example().metadata
    texts = text.split("\n")
    return [Document(text=text, metadata={"text": text}) for text in texts]


@pytest.fixture(scope="session")
def nodes(documents, embed_model) -> List[TextNode]:
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=1024, chunk_overlap=200),
            embed_model,
        ],
    )
    return pipeline.run(documents=documents)


DB_NAME = os.environ.get("MONGODB_DATABASE", "llama_index_test_db")
collection_name = os.environ.get("MONGODB_COLLECTION", "llama_index_test_vectorstore")
vector_index_name = os.environ.get("MONGODB_INDEX", "vector_index")
MONGODB_URI = os.environ.get("MONGODB_URI")


@pytest.fixture(scope="session")
def atlas_client() -> MongoClient:
    if MONGODB_URI is None:
        raise pytest.skip("Requires MONGODB_URI in os.environ")

    client = MongoClient(MONGODB_URI)
    assert DB_NAME in client.list_database_names()
    return client


@pytest.fixture()
def vector_store(
    atlas_client: MongoClient, embed_model: OpenAIEmbedding
) -> MongoDBAtlasVectorSearch:
    # Set up the default llm to be used in tests.
    if isinstance(embed_model, AzureOpenAIEmbedding):
        deployment_name = os.environ.get("AZURE_LLM_DEPLOYMENT", "gpt-4o-mini")
        Settings.llm = AzureOpenAI(
            engine=deployment_name, api_key=os.environ["AZURE_OPENAI_API_KEY"]
        )
    return MongoDBAtlasVectorSearch(
        mongodb_client=atlas_client,
        db_name=DB_NAME,
        collection_name=collection_name,
        vector_index_name=vector_index_name,
        fulltext_index_name="fulltext_index",
    )
