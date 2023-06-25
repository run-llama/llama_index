from llama_index import ListIndex
from llama_index.storage.docstore import RedisDocumentStore
from llama_index.storage.index_store import RedisIndexStore
from llama_index import StorageContext, load_index_from_storage, Document

import pdb

pdb.set_trace()
ds = RedisDocumentStore.from_host_and_port("127.0.0.1", "6379", namespace="data4")
idxs = RedisIndexStore.from_host_and_port("127.0.0.1", "6379", namespace="data4")

storage_context = StorageContext.from_defaults(docstore=ds, index_store=idxs)

import os

os.environ["OPENAI_API_KEY"] = "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"

index = ListIndex.from_documents(
    [Document("hello world2")], storage_context=storage_context
)

index.docstore.docs
