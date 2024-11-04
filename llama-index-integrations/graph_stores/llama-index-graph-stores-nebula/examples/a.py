from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.llm = OpenAI(model="gpt-4o", temperature=0.3)
Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")
# Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")

from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
from llama_index.graph_stores.nebula import NebulaPropertyGraphStore

graph_store = NebulaPropertyGraphStore(
    space="llamaindex_nebula_property_graph", overwrite=True
)

from llama_index.core.vector_stores.simple import SimpleVectorStore

vec_store = SimpleVectorStore()
# vec_store = SimpleVectorStore.from_persist_path("./vec_store.json")

from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.storage.storage_context import StorageContext

index = PropertyGraphIndex.from_documents(
    documents,
    property_graph_store=graph_store,
    vector_store=vec_store,
    show_progress=True,
)

index.storage_context.vector_store.persist("./vec_store.json")
query = "who is Paul Graham?"
retrieved = index.as_retriever().retrieve(query)
answer = index.as_query_engine().query(query)
print(retrieved, answer)
