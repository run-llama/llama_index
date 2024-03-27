# Simple Website Loader

```bash
pip install llama-index-readers-vedastro
```

This loader is a simple astrology prediction text generator from given birth time.

## Usage

To use this loader, you need to pass in birth time exp : "Location/Delhi,India/Time/01:30/14/02/2024/+05:30".

```python
from llama_index.readers.web import SimpleBirthTimeReader

loader = SimpleBirthTimeReader()
documents = loader.load_data("Location/Delhi,India/Time/01:30/14/02/2024/+05:30")
```

## Examples

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.

### LlamaIndex

```python
from llama_index.core import VectorStoreIndex, download_loader

from llama_index.readers.vedastro import SimpleBirthTimeReader

loader = SimpleBirthTimeReader()
documents = loader.load_data("Location/Delhi,India/Time/01:30/14/02/2024/+05:30")
index = VectorStoreIndex.from_documents(documents)
index.query("Describe my love life")
```

### LangChain

Note: Make sure you change the description of the `Tool` to match your use-case.

```python
from llama_index.core import VectorStoreIndex, download_loader
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory

from llama_index.readers.vedastro import SimpleBirthTimeReader

loader = SimpleBirthTimeReader()
documents = loader.load_data("Location/Delhi,India/Time/01:30/14/02/2024/+05:30")
index = VectorStoreIndex.from_documents(documents)

tools = [
    Tool(
        name="Horoscope Index",
        func=lambda q: index.query(q),
        description=f"Useful when you want answer questions about the prediction text about a person.",
    ),
]
llm = OpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history")
agent_chain = initialize_agent(
    tools, llm, agent="zero-shot-react-description", memory=memory
)

output = agent_chain.run(input="Describe my love life")
```
