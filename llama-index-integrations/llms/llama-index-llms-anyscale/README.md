# LlamaIndex Llms Integration: Anyscale

### Installation

```bash
%pip install llama-index-llms-anyscale
!pip install llama-index
```

### Basic Usage

```py
from llama_index.llms.anyscale import Anyscale
from llama_index.core.llms import ChatMessage

# Call chat with ChatMessage List
# You need to either set env var ANYSCALE_API_KEY or set api_key in the class constructor

# Example of setting API key through environment variable
# import os
# os.environ['ANYSCALE_API_KEY'] = '<your-api-key>'

# Initialize the Anyscale LLM with your API key
llm = Anyscale(api_key="<your-api-key>")

# Chat Example
message = ChatMessage(role="user", content="Tell me a joke")
resp = llm.chat([message])
print(resp)

# Expected Output:
# assistant: Sure, here's a joke for you:
#
# Why couldn't the bicycle stand up by itself?
#
# Because it was two-tired!
#
# I hope that brought a smile to your face! Is there anything else I can assist you with?
```

### Streaming Example

```py
message = ChatMessage(role="user", content="Tell me a story in 250 words")
resp = llm.stream_chat([message])
for r in resp:
    print(r.delta, end="")

# Output Example:
# Once upon a time, there was a young girl named Maria who lived in a small village surrounded by lush green forests.
# Maria was a kind and gentle soul, loved by everyone in the village. She spent most of her days exploring the forests,
# discovering new species of plants and animals, and helping the villagers with their daily chores...
# (Story continues until it reaches the word limit.)
```

### Completion Example

```py
resp = llm.complete("Tell me a joke")
print(resp)

# Expected Output:
# assistant: Sure, here's a joke for you:
#
# Why couldn't the bicycle stand up by itself?
#
# Because it was two-tired!
```

### Streaming Completion Example

```py
resp = llm.stream_complete("Tell me a story in 250 words")
for r in resp:
    print(r.delta, end="")

# Example Output:
# Once upon a time, there was a young girl named Maria who lived in a small village...
# (Stream continues as the story is generated.)
```

### Model Configuration

```py
llm = Anyscale(model="codellama/CodeLlama-34b-Instruct-hf")
resp = llm.complete("Show me the c++ code to send requests to HTTP Server")
print(resp)
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/anyscale/
