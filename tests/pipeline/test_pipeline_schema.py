from llama_index.prompts import Prompt
from llama_index import OpenAIEmbedding
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.llms import OpenAI


def test_llm_schema() -> None:
    llm = OpenAI("text-davinci-003")
    schema = llm.schema()
    assert (
        schema.json()
        == """\
{
  "name": "LLM",
  "metadata": {
    "context_window": 4097,
    "num_output": -1,
    "is_chat_model": false,
    "is_function_calling_model": false,
    "model_name": "text-davinci-003"
  },
  "children": [],
  "inputs": []
}"""
    )


def test_prompt_schema() -> None:
    prompt = Prompt(
        template="This is a {{my_var}} prompt",
    )
    schema = prompt.schema()
    assert (
        schema.json()
        == """\
{
  "name": "Prompt",
  "metadata": {
    "type": "custom",
    "template": "This is a {{my_var}} prompt",
    "input_variables": []
  },
  "children": [],
  "inputs": []
}"""
    )


def test_embedding_schema() -> None:
    embedding = OpenAIEmbedding()
    schema = embedding.schema()
    assert (
        schema.json()
        == """\
{
  "name": "Embedding",
  "metadata": {
    "batch_size": 10,
    "embedding_class": "OpenAIEmbedding"
  },
  "children": [],
  "inputs": []
}"""
    )


def test_document_store_schema() -> None:
    document_store = SimpleDocumentStore()
    schema = document_store.schema()
    assert (
        schema.json()
        == """\
{
  "name": "DocumentStore",
  "metadata": {
    "doc_store_class": "SimpleDocumentStore",
    "num_documents": 0
  },
  "children": [],
  "inputs": []
}"""
    )


def test_vector_index_schema() -> None:
    pass


def test_retriever_query_engine_schema() -> None:
    pass
    # index = VectorStoreIndex.from_documents([Document.example()])


def test_prompt_schema_partial_format() -> None:
    pass


def test_prompt_schema_prompt_selector() -> None:
    pass
