from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ResponseSynthesizer,
    DocumentSummaryIndex,
    LLMPredictor,
    ServiceContext,
    KeywordTableIndex,
    KnowledgeGraphIndex,
    ListIndex,
    MockLLMPredictor,
    TreeIndex,
)
import os
import openai
from langchain.chat_models import ChatOpenAI
from llama_index.storage.storage_context import StorageContext
from langchain import OpenAI
from llama_index.graph_stores import SimpleGraphStore

# Set environment variable
os.environ['OPENAI_API_KEY'] = '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'
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load documents
documents = SimpleDirectoryReader("../../../docs/examples/data/paul_graham").load_data()

# ### VectorStoreIndex

print("\nVectorStoreIndex with show_progress=True\n")
VectorStoreIndex.from_documents(documents, show_progress=True)


# ### DocumentSummaryIndex

llm_predictor_chatgpt = LLMPredictor(
    llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
)
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor_chatgpt, chunk_size=1024
)

print("\nDocumentSummaryIndex with show_progress=True\n")
response_synthesizer = ResponseSynthesizer.from_args(
    response_mode="tree_summarize", use_async=True
)
DocumentSummaryIndex.from_documents(
    documents,
    service_context=service_context,
    response_synthesizer=response_synthesizer,
    show_progress=True,
)

print("\nDocumentSummaryIndex with show_progress=False\n")
DocumentSummaryIndex.from_documents(
    documents,
    service_context=service_context,
    response_synthesizer=response_synthesizer,
    show_progress=False,
)

# ### KeywordTableIndex

print("\nKeywordTableIndex with show_progress=True, use_async=True\n")
KeywordTableIndex.from_documents(
    documents=documents, show_progress=True, use_async=True
)

print("\nKeywordTableIndex with show_progress=True, use_async=False\n")
KeywordTableIndex.from_documents(
    documents=documents, show_progress=True, use_async=False
)

print("\nKeywordTableIndex with show_progress=False, use_async=True\n")
KeywordTableIndex.from_documents(documents=documents, use_async=True)

print("\nKeywordTableIndex with show_progress=False, use_async=False\n")
KeywordTableIndex.from_documents(documents=documents)

# ### KnowledgeGraphIndex

print("\nKnowledgeGraphIndex with show_progress=True, use_async=False\n")
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-002"))
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, chunk_size=512
)
graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)
KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=2,
    storage_context=storage_context,
    service_context=service_context,
    show_progress=True,
    use_async=False,
)
print("\nKnowledgeGraphIndex with show_progress=True, use_async=True\n")
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-002"))
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, chunk_size=512
)
graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)
KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=2,
    storage_context=storage_context,
    service_context=service_context,
    show_progress=True,
    use_async=True,
)

# ### ListIndex

print("\nListIndex with show_progress=True\n")
ListIndex.from_documents(documents=documents, show_progress=True)

print("\nListIndex with show_progress=False\n")
ListIndex.from_documents(documents=documents)

# ### TreeIndex

print("\nTreeIndex with show_progress=True,  use_async=True\n")
llm_predictor = MockLLMPredictor(max_tokens=256)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
TreeIndex.from_documents(
    documents, service_context=service_context, show_progress=True, use_async=True
)

print("\nTreeIndex with show_progress=True, use_async=False\n")
TreeIndex.from_documents(
    documents, service_context=service_context, show_progress=True, use_async=False
)

print("\nTreeIndex with show_progress=False, use_async=True\n")
TreeIndex.from_documents(documents, service_context=service_context, use_async=True)

print("\nTreeIndex with show_progress=False, use_async=False\n")
TreeIndex.from_documents(documents, service_context=service_context)



