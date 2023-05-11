import os
os.environ["OPENAI_API_KEY"] = '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'

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
documents = SimpleDirectoryReader('data').load_data()
index = GPTVectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
# query_engine.query("what is AI?")
# index.storage_context.persist()
while True:
    question_prompt = input("Your question: ")
    response = query_engine.query(question_prompt)
    print("Answer: ", response)
#
#