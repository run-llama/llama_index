from llama_index.prompts import Prompt
from llama_index.schema import Document
from llama_index import OpenAIEmbedding
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.llms import OpenAI

def test_llm_schema():
    llm = OpenAI('text-davinci-003')
    schema = llm.schema()
    assert schema.json() == """\
{
  "name": "LLM",
  "metadata": {
    "context_window": 4097,
    "num_output": -1,
    "is_chat_model": false,
    "is_function_calling_model": false,
    "model_name": "text-davinci-003"
  },
  "children": []
}"""
  
def test_prompt_schema():
    prompt = Prompt(
        template="This is a {{my_var}} prompt",
    )
    schema = prompt.schema()
    assert schema.json() == """\
{
  "name": "Prompt",
  "metadata": {
    "type": "custom",
    "template": "This is a {{my_var}} prompt",
    "input_variables": []
  },
  "children": []
}"""

def test_embedding_schema():
    embedding = OpenAIEmbedding()
    schema = embedding.schema()
    assert schema.json() == """\
{
  "name": "Embedding",
  "metadata": {
    "batch_size": 10,
    "embedding_class": "OpenAIEmbedding"
  },
  "children": []
}"""

def test_document_store_schema():
    document_store = SimpleDocumentStore()
    schema = document_store.schema()
    assert schema.json() == """\
{
  "name": "DocumentStore",
  "metadata": {
    "doc_store_class": "SimpleDocumentStore",
    "num_documents": 0
  },
  "children": []
}"""

def test_vector_index_schema():
    pass

def test_retriever_query_engine_schema():
    pass
    #index = VectorStoreIndex.from_documents([Document.example()])

def test_prompt_schema_partial_format():
    pass

def test_prompt_schema_prompt_selector():
    pass