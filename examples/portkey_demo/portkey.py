from llama_index.llms import Portkey
from llama_index.llms import ChatMessage
from rubeus import LLMBase

# openai_llm = LLMBase(provider="openai", model="gpt-3.5-turbo")
# llm = Portkey(mode="fallback", api_key="").add_llms(llm_params=[openai_llm])

# messages = [
#     ChatMessage(role="system", content="You are a pirate with a colorful personality"),
#     ChatMessage(role="user", content="What is your name"),
# ]
# response = llm.chat(messages)
# print(response)

openai_llm = LLMBase(provider="openai", model="gpt-3.5-turbo")
llm = Portkey(
    mode="fallback", cache_status="simple", metadata={"_user": "noble-varghese"}
).add_llms(llm_params=[openai_llm])

print("Testing stream chat functionality:")
stream_messages = [
    ChatMessage(role="system", content="You are a helpful assistant"),
    ChatMessage(role="user", content="What can you do?"),
]
stream_chat_response = llm.stream_chat(stream_messages)
for resp in stream_chat_response:
    print(resp.delta, end=" ", flush=True)
# # Test stream completion functionality
# print("Testing stream completion functionality:")
# stream_completion_response = llm.stream_complete("In a far away land")
# print(stream_completion_response)
# for resp in stream_completion_response:
#     print(resp)
# print()
