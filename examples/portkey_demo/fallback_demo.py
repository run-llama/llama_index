import os
from llama_index.llms import Portkey
from llama_index.llms import ChatMessage  # We'll use this later
from portkey import LLMOptions

os.environ["PORTKEY_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""


metadata = {
    "_environment": "production",
    "_prompt": "test",
    "_user": "user-1234",
    "_organisation": "llama_index",
}

# Define the Portkey interface.
pk_client = Portkey(mode="fallback")

# Define the provider spec in the LLMBase spec. Customise the LLMs as per requirement.
openai_llm = LLMOptions(
    provider="openai",
    model="gpt-4",
    trace_id="portkey_llamaindex",
    metadata=metadata,
    virtual_key="open-ai-key-66ah788",
)
anthropic_llm = LLMOptions(
    provider="openai",
    model="claude-2",
    max_tokens=256,
    trace_id="portkey_llamaindex",
    metadata=metadata,
    virtual_key="anthropic-key-351feb",
)

pk_client.add_llms([openai_llm, anthropic_llm])

messages = [
    ChatMessage(role="system", content="You are a helpful assistant"),
    ChatMessage(role="user", content="What can you do?"),
]

print("Testing Portkey Llamaindex integration:")
response = pk_client.chat(messages)
print(response)
