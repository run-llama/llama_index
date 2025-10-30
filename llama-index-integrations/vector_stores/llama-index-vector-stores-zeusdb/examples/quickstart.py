# quickstart.py
from dotenv import load_dotenv

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.zeusdb import ZeusDBVectorStore

load_dotenv()  # reads .env and sets OPENAI_API_KEY

# Set up embedding model
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-5")

# Create ZeusDB vector store
vector_store = ZeusDBVectorStore(
    dim=1536,  # OpenAI embedding dimension
    distance="cosine",
    index_type="hnsw",
)

# Create storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create documents
documents = [
    Document(text="ZeusDB is a high-performance vector database."),
    Document(text="LlamaIndex provides RAG capabilities."),
    Document(text="Vector search enables semantic similarity."),
]

# Create index and store documents
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Query the index
query_engine = index.as_query_engine()
response = query_engine.query("What is ZeusDB?")
print(response)
