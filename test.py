import os
import ipdb
from azure.identity import AzureCliCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
)
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore, IndexManagement


# Environment Variables
AZURE_OPENAI_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT", "https://llm-openai-north-datanalytics-dev.openai.azure.com/"
)
AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME = os.getenv(
    "AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME", "gpt-4o"
)
AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME", "text-embedding-3-large"
)
SEARCH_SERVICE_ENDPOINT = os.getenv(
    "AZURE_SEARCH_SERVICE_ENDPOINT", "https://search-dnacompliancebot.search.windows.net"
)
INDEX_NAME = "index-dnacompliancebot"
CONVERSATION_INDEX = "conversations"

credentials = AzureCliCredential()
cognitive_services_specific_ad_token = get_bearer_token_provider(
    credentials, "https://cognitiveservices.azure.com/.default"
)

# Initialize Azure OpenAI and embedding models
llm = AzureOpenAI(
    model=AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME,
    deployment_name=AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME,
    use_azure_ad=True,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-10-21",
)

embed_model = AzureOpenAIEmbedding(
    model=AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME,
    deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME,
    use_azure_ad=True,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-10-21",
)

semantic_search_config = SemanticSearch(
    configurations=[
        SemanticConfiguration(
            name="default",
            prioritized_fields=SemanticPrioritizedFields(
                content_fields=[SemanticField(field_name="content")],
            ),
        )
    ]
)
# Initialize search clients
index_client = SearchIndexClient(
    endpoint=SEARCH_SERVICE_ENDPOINT, credential=credentials
)
search_client = SearchClient(endpoint=SEARCH_SERVICE_ENDPOINT, index_name=INDEX_NAME, credential=credentials)

from llama_index.core.settings import Settings

Settings.llm = llm
Settings.embed_model = embed_model

# Initialize the vector store
vector_store_1 = AzureAISearchVectorStore(
    search_or_index_client=index_client,
    index_name=INDEX_NAME,
    index_management=IndexManagement.VALIDATE_INDEX,
    id_field_key="id",
    chunk_field_key="content",
    embedding_field_key="content_vector",
    embedding_dimensionality=3072,
    metadata_string_field_key="metadata",
    doc_id_field_key="title",
    language_analyzer="en.lucene",
    vector_algorithm_type="hnsw",
)

vector_store = AzureAISearchVectorStore(
    search_or_index_client=index_client,
    index_name=INDEX_NAME,
    index_management=IndexManagement.VALIDATE_INDEX,
    id_field_key="id",
    chunk_field_key="content",
    embedding_field_key="content_vector",
    embedding_dimensionality=3072,
    metadata_string_field_key="metadata",
    doc_id_field_key="title",
    language_analyzer="en.lucene",
    vector_algorithm_type="hnsw",
    semantic_config_name="default",
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
storage_context_1 = StorageContext.from_defaults(vector_store=vector_store_1)
index = VectorStoreIndex.from_documents(
    [],
    storage_context=storage_context,
)
index_1 = VectorStoreIndex.from_documents(
    [],
    storage_context=storage_context_1,
)

from llama_index.core.schema import MetadataMode
query = "Hi Chat, can you tell me what is our current data processing agreement with AWS?"
query_engine = index.as_query_engine(llm=llm, similarity_top_k=3, response_mode="tree_summarize")
response = query_engine.query(query)
print(response)

query_engine_1 = index_1.as_query_engine(llm=llm, similarity_top_k=3, response_mode="tree_summarize")
response_1 = query_engine_1.query(query)

ipdb.set_trace()

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core import get_response_synthesizer

# define response synthesizer
response_synthesizer = get_response_synthesizer()

semantic_hybrid_retriever = index.as_retriever(
    vector_store_query_mode=VectorStoreQueryMode.SEMANTIC_HYBRID, similarity_top_k=5
)

ipdb.set_trace()

semantic_hybrid_query_engine = RetrieverQueryEngine(
    retriever=semantic_hybrid_retriever, response_synthesizer=response_synthesizer
)

print(semantic_hybrid_query_engine.query("Hi Chat, can you tell me what is our current data processing agreement with AWS?"))

