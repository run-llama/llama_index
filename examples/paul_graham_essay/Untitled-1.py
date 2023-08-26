from llama_index import StorageContext, load_index_from_storage
from llama_index.llms import OpenAI
from llama_index import ServiceContext

# documents = SimpleDirectoryReader('./data').load_data()

# print('Setting the LLm context')
# llm = OpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=256)
# service_context = ServiceContext.from_defaults(llm=llm)

# print('indexing it.')
# index = KeywordTableIndex.from_documents(
#     documents=documents, service_context=service_context
# )


# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="./storage")
# load index
index = load_index_from_storage(storage_context)

llm = OpenAI(
    model="text-davinci-002",
    additional_kwargs={
        "api_base": "https://api.portkey.ai/v1/proxy",
        "headers": {
            "x-portkey-api-key": "/turdjWE+tIUeAzmzGxGEkkJLBQ=",
            "x-portkey-mode": "proxy openai",
        },
    },
)
service_context = ServiceContext.from_defaults(llm=llm)

print("querying the index.")
query_engine = index.as_query_engine(service_context=service_context)
res = query_engine.query("What did the author do after his time at Y Combinator?")
print("\n response: ", res)


# index.storage_context.persist(persist_dir='./storage')


# from llama_index import StorageContext, load_index_from_storage

# # rebuild storage context
# storage_context = StorageContext.from_defaults(persist_dir='./storage')
# # load index
# index = load_index_from_storage(storage_context)
