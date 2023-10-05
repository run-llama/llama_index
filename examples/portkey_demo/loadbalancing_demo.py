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
    "_user": "user",
    "_organisation": "acme",
}

# Define the Portkey interface.
pk_client = Portkey(mode="ab_test")

# Define the provider spec in the LLMOptions spec. Customise the LLMs as per requirement.
# Added here are the weights that specify a 40:60 split in the requests.
openai_llm = LLMOptions(
    provider="openai",
    model="gpt-4",
    weight=0.4,
    metadata=metadata,
    virtual_key="open-ai-key-66ah788",
)
anthropic_llm = LLMOptions(
    provider="openai",
    model="claude-2",
    max_tokens=256,
    weight=0.6,
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
