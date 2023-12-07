import os

from llama_index.llms import MonsterLLM
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext

deploy_llm = MonsterLLM(model="deploy-llm", base_url = "https://216.153.50.231", monster_api_key="1db5f30d-b77b-4187-bc67-8148414b55bc", temperature=0.75)

print(deploy_llm.complete("What is Retrieval-Augmented Generation?"))
for i in deploy_llm.stream_complete("What is Retrieval-Augmented Generation?"):
    print(i)