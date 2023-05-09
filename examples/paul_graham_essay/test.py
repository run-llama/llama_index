import os
os.environ["OPENAI_API_KEY"] = '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'

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
documents = SimpleDirectoryReader('data').load_data()
index = GPTVectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
# query_engine.query("what is AI?")
# index.storage_context.persist()
response = query_engine.query("What did the author do growing up?")
print(response)
#