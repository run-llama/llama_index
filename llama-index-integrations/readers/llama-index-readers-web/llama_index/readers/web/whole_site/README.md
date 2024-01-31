# WholeSiteReader

The WholeSiteReader is a sophisticated web scraping tool that employs a breadth-first search (BFS) algorithm. It's designed to methodically traverse and extract content from entire websites, focusing specifically on predefined URL paths.

## Features

- **Breadth-First Search (BFS):** Traverses a website thoroughly, ensuring comprehensive coverage of all accessible pages.
- **Depth Control:** Limits scraping to a specified depth within a site's structure.
- **URL Prefix Focus:** Targets scraping efforts to specific subsections of a site based on URL prefixes.
- **Selenium-Based:** Leverages Selenium for dynamic interaction with web pages, supporting JavaScript-rendered content.

```python
from llama_index import download_loader

WholeSiteReader = download_loader("WholeSiteReader")
# Initialize the scraper with a prefix URL and maximum depth
scraper = WholeSiteReader(
    prefix="https://www.paulgraham.com/", max_depth=10  # Example prefix
)

# Start scraping from a base URL
documents = scraper.load_data(
    base_url="https://www.paulgraham.com/articles.html"
)  # Example base URL
```

## Examples

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.

### LlamaIndex

```python
from llama_index import VectorStoreIndex, download_loader

WholeSiteReader = download_loader("WholeSiteReader")

# Initialize the scraper with a prefix URL and maximum depth
scraper = WholeSiteReader(
    prefix="https://docs.llamaindex.ai/en/stable/",  # Example prefix
    max_depth=10,
)

# Start scraping from a base URL
documents = scraper.load_data(
    base_url="https://docs.llamaindex.ai/en/stable/"
)  # Example base URL
index = VectorStoreIndex.from_documents(documents)
index.query("What language is on this website?")
```

### LangChain

Note: Make sure you change the description of the `Tool` to match your use-case.

```python
from llama_index import VectorStoreIndex, download_loader
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory

WholeSiteReader = download_loader("WholeSiteReader")

# Initialize the scraper with a prefix URL and maximum depth
scraper = WholeSiteReader(
    prefix="https://docs.llamaindex.ai/en/stable/",  # Example prefix
    max_depth=10,
)

# Start scraping from a base URL
documents = scraper.load_data(
    base_url="https://docs.llamaindex.ai/en/stable/"
)  # Example base URL
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
