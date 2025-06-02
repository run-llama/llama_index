from llama_index.core import Document, VectorStoreIndex, set_global_handler

# All configuration arguments are optional. However, if you don't have api_key and url you should
# provide their values as environment variables : LITERAL_API_KEY, LITERAL_API_URL
set_global_handler(
    "literalai",
    # api_key="lsk_xxx",
    # url="https://cloud.getliteral.ai",
    # batch_size=5,
    # environment=None,
    # disabled=False
)

# This example uses OpenAI by default so don't forget to set an OPENAI_API_KEY
index = VectorStoreIndex.from_documents([Document.example()])
query_engine = index.as_query_engine()

questions = [
    "Tell me about LLMs",
    "How do you fine-tune a neural network ?",
    "What is RAG ?",
]

for question in questions:
    print(f"> \033[92m{question}\033[0m")
    response = query_engine.query(question)
    print(response)
