from llama_index.llms import Portkey
from llama_index.llms import ChatMessage
from llama_index.llms.portkey_utils import LLMBase

openai_llm = LLMBase(provider="openai", model="gpt-3.5-turbo")
llm = Portkey(mode="fallback", api_key="").add_llms(llm_params=[openai_llm])

messages = [
    ChatMessage(role="system", content="You are a pirate with a colorful personality"),
    ChatMessage(role="user", content="What is your name"),
]
response = llm.chat(messages)
print(response)
