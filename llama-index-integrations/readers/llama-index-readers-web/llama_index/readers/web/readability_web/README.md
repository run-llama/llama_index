# Readability Webpage Loader

```bash
pip install llama-index-readers-web
```

Extracting relevant information from a fully rendered web page.
During the processing, it is always assumed that web pages used as data sources contain textual content.

It is particularly effective for websites that use client-side rendering.

1. Load the page and wait for it rendered. (playwright)
2. Inject Readability.js to extract the main content.

## Usage

To use this loader, you need to pass in a single of URL.

```python
from llama_index.readers.web import ReadabilityWebPageReader

# or set proxy server for playwright: loader = ReadabilityWebPageReader(proxy="http://your-proxy-server:port")
# For some specific web pages, you may need to set "wait_until" to "networkidle". loader = ReadabilityWebPageReader(wait_until="networkidle")
loader = ReadabilityWebPageReader()

documents = loader.load_data(
    url="https://support.squarespace.com/hc/en-us/articles/206795137-Pages-and-content-basics"
)
```

## Examples

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).

### LlamaIndex

```python
from llama_index.readers.web import ReadabilityWebPageReader

loader = ReadabilityWebPageReader()
documents = loader.load_data(
    url="https://support.squarespace.com/hc/en-us/articles/206795137-Pages-and-content-basics"
)

index = VectorStoreIndex.from_documents(documents)
print(index.query("What is pages?"))
```

### LangChain

Note: Make sure you change the description of the `Tool` to match your use-case.

```python
from llama_index.core import VectorStoreIndex, download_loader
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory

from llama_index.readers.web import ReadabilityWebPageReader

loader = ReadabilityWebPageReader()
documents = loader.load_data(
    url="https://support.squarespace.com/hc/en-us/articles/206795137-Pages-and-content-basics"
)

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

output = agent_chain.run(input="What is pages?")
```
