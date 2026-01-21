from fastapi import FastAPI
from pydantic import BaseModel

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

app = FastAPI(title="LlamaIndex FastAPI RAG (Ollama)")

# Configure local LLM and embedding model via Ollama
Settings.llm = Ollama(model="llama3")
Settings.embed_model = OllamaEmbedding(model_name="llama3")

# Load documents and build index at startup
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_documents(request: QueryRequest):
    # Query indexed documents using a local LLM via Ollama.
    response = query_engine.query(request.query)
    return {"response": str(response)}
