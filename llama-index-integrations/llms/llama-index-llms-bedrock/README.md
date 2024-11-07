# LlamaIndex Llms Integration: Bedrock

### Installation

```bash
%pip install llama-index-llms-bedrock
!pip install llama-index
```

### Basic Usage

```py
from llama_index.llms.bedrock import Bedrock

# Set your AWS profile name
profile_name = "Your aws profile name"

# Simple completion call
resp = Bedrock(
    model="amazon.titan-text-express-v1", profile_name=profile_name
).complete("Paul Graham is ")
print(resp)

# Expected output:
# Paul Graham is a computer scientist and entrepreneur, best known for co-founding
# the Silicon Valley startup incubator Y Combinator. He is also a prominent writer
# and speaker on technology and business topics...
```

### Call chat with a list of messages

```py
from llama_index.core.llms import ChatMessage
from llama_index.llms.bedrock import Bedrock

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="Tell me a story"),
]

resp = Bedrock(
    model="amazon.titan-text-express-v1", profile_name=profile_name
).chat(messages)
print(resp)

# Expected output:
# assistant: Alright, matey! Here's a story for you: Once upon a time, there was a pirate
# named Captain Jack Sparrow who sailed the seas in search of his next adventure...
```

### Streaming

#### Using stream_complete endpoint

```py
from llama_index.llms.bedrock import Bedrock

llm = Bedrock(model="amazon.titan-text-express-v1", profile_name=profile_name)
resp = llm.stream_complete("Paul Graham is ")
for r in resp:
    print(r.delta, end="")

# Expected Output (Stream):
# Paul Graham is a computer programmer, entrepreneur, investor, and writer, best known
# for co-founding the internet firm Y Combinator...
```

### Streaming chat

```py
from llama_index.llms.bedrock import Bedrock

llm = Bedrock(model="amazon.titan-text-express-v1", profile_name=profile_name)
messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="Tell me a story"),
]
resp = llm.stream_chat(messages)
for r in resp:
    print(r.delta, end="")

# Expected Output (Stream):
# Once upon a time, there was a pirate with a colorful personality who sailed the
# high seas in search of adventure...
```

### Configure Model

```py
from llama_index.llms.bedrock import Bedrock

llm = Bedrock(model="amazon.titan-text-express-v1", profile_name=profile_name)
resp = llm.complete("Paul Graham is ")
print(resp)

# Expected Output:
# Paul Graham is a computer scientist, entrepreneur, investor, and writer. He co-founded
# Viaweb, the first commercial web browser...
```

### Connect to Bedrock with Access Keys

```py
from llama_index.llms.bedrock import Bedrock

llm = Bedrock(
    model="amazon.titan-text-express-v1",
    aws_access_key_id="AWS Access Key ID to use",
    aws_secret_access_key="AWS Secret Access Key to use",
    aws_session_token="AWS Session Token to use",
    region_name="AWS Region to use, e.g. us-east-1",
)

resp = llm.complete("Paul Graham is ")
print(resp)

# Expected Output:
# Paul Graham is an American computer scientist, entrepreneur, investor, and author,
# best known for co-founding Viaweb, the first commercial web browser...
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/bedrock/
