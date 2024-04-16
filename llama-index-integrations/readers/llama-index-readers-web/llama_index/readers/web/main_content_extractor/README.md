# MainContentExtractor Website Loader

```bash
pip install llama-index-readers-web
```

This loader is a web scraper that fetches the text from static websites using the `MainContentExtractor` Python package.

For information on how to extract main content, README in the following github repository

[HawkClaws/main_content_extractor](https://github.com/HawkClaws/main_content_extractor)

## Usage

To use this loader, you need to pass in an array of URLs.

```python
from llama_index.readers.web import MainContentExtractorReader

loader = MainContentExtractorReader()
documents = loader.load_data(urls=["https://google.com"])
```

## Examples

### LlamaIndex

```python
from llama_index.core import VectorStoreIndex, download_loader

from llama_index.readers.web import MainContentExtractorReader

loader = MainContentExtractorReader()
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

from llama_index.readers.web import MainContentExtractorReader

loader = MainContentExtractorReader()
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
