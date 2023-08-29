from llama_index.storage import StorageContext
from llama_index import load_index_from_storage
from llama_index.llms import Portkey
from llama_index import ServiceContext
from llama_index.llms import ChatMessage
from llama_index.llms.portkey_utils import LLMBase, ProviderTypes

openai_llm = LLMBase(provider="openai", model="gpt-3.5-turbo")
llm = Portkey(mode="fallback", api_key="").add_llms(llm_params=[openai_llm])

messages = [
    ChatMessage(role="system", content="You are a pirate with a colorful personality"),
    ChatMessage(role="user", content="What is your name"),
]
response = llm.chat(messages)
print(response)
