# Trafilatura Website Loader

```bash
pip install llama-index-readers-web
```

This loader is a web scraper that fetches the text from static websites using the `trafilatura` Python package.

## Usage

To use this loader, you need to pass in an array of URLs.

```python
from llama_index.readers.web import TrafilaturaWebReader

loader = TrafilaturaWebReader()
documents = loader.load_data(urls=["https://google.com"])
```

### Additional Parameters

You can also pass in additional parameters to the `load_data` function.

Most of the functions follow the original `trafilatura.extract` API. You can find more information [here](https://trafilatura.readthedocs.io/en/latest/corefunctions.html#extract).

```python
documents = loader.load_data(urls=["https://google.com"], favor_recall=True)
```

## Examples

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).

### LlamaIndex

```python
from llama_index.core import VectorStoreIndex, download_loader

from llama_index.readers.web import TrafilaturaWebReader

loader = TrafilaturaWebReader()
documents = loader.load_data(urls=["https://google.com"])
index = VectorStoreIndex.from_documents(documents)
index.query("What language is on this website?")
```

### LangChain

Note: Make sure you change the description of the `Tool` to match your use-case.

```python
from llama_index.core import VectorStoreIndex, download_loader
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory

from llama_index.readers.web import TrafilaturaWebReader

loader = TrafilaturaWebReader()
documents = loader.load_data(urls=["https://google.com"])
index = VectorStoreIndex.from_documents(documents)

tools = [
    Tool(
        name="Website Index",
        func=lambda q: index.query(q),
        description=f"Useful when you want answer questions about the text on websites.",
    ),
]
llm = OpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history")
agent_chain = initialize_agent(
    tools, llm, agent="zero-shot-react-description", memory=memory
)

output = agent_chain.run(input="What language is on this website?")
```
