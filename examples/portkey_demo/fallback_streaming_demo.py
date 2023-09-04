import os
from llama_index.llms import Portkey
from llama_index.llms import ChatMessage  # We'll use this later
from rubeus import LLMBase

os.environ["PORTKEY_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""


metadata = {
    "_environment": "production",
    "_prompt": "test",
    "_user": "user",
    "_organisation": "acme",
}

# Define the Portkey interface.
pk_llm = Portkey(mode="fallback", trace_id="portkey_llamaindex", metadata=metadata)

# Define the provider spec in the LLMBase spec. Customise the LLMs as per requirement.
openai_llm = LLMBase(provider="openai", model="gpt-4")
anthropic_llm = LLMBase(provider="openai", model="claude-2", max_tokens=256)

pk_llm.add_llms([openai_llm, anthropic_llm])

messages = [
    ChatMessage(role="system", content="You are a helpful assistant"),
    ChatMessage(role="user", content="What can you do?"),
]

print("Testing Portkey Llamaindex integration:")
response = pk_llm.stream_chat(messages)

for i in response:
    print(i.delta, end="", flush=True)
