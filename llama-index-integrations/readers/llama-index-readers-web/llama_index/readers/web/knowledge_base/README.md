# Knowledge Base Website Loader

```bash
pip install llama-index-readers-web
```

This loader is a web crawler and scraper that fetches text content from websites hosting public knowledge bases. Examples are the [Intercom help center](https://www.intercom.com/help/en/) or the [Robinhood help center](https://robinhood.com/us/en/support/). Typically these sites have a directory structure with several sections and many articles in each section. This loader crawls and finds all links that match the article path provided, and scrapes the content of each article. This can be used to create bots that answer customer questions based on public documentation.

It uses [Playwright](https://playwright.dev/python/) to drive a browser. This reduces the chance of getting blocked by Cloudflare or other CDNs, but makes it a bit more challenging to run on cloud services.

## Usage

First run

```
playwright install
```

This installs the browsers that Playwright requires.

To use this loader, you need to pass in the root URL and the string to search for in the URL to tell if the crawler has reached an article. You also need to pass in several CSS selectors so the cralwer knows which links to follow and which elements to extract content from. use

```python
from llama_index.readers.web import KnowledgeBaseWebReader

loader = KnowledgeBaseWebReader()
documents = loader.load_data(
    root_url="https://www.intercom.com/help",
    link_selectors=[".article-list a", ".article-list a"],
    article_path="/articles",
    body_selector=".article-body",
    title_selector=".article-title",
    subtitle_selector=".article-subtitle",
)
```

## Examples

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).

### LlamaIndex

```python
from llama_index.core import VectorStoreIndex, download_loader

from llama_index.readers.web import KnowledgeBaseWebReader

loader = KnowledgeBaseWebReader()
documents = loader.load_data(
    root_url="https://support.intercom.com",
    link_selectors=[".article-list a", ".article-list a"],
    article_path="/articles",
    body_selector=".article-body",
    title_selector=".article-title",
    subtitle_selector=".article-subtitle",
)
index = VectorStoreIndex.from_documents(documents)
index.query("What languages does Intercom support?")
```

### LangChain

Note: Make sure you change the description of the `Tool` to match your use-case.

```python
from llama_index.core import VectorStoreIndex, download_loader
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory

from llama_index.readers.web import KnowledgeBaseWebReader

loader = KnowledgeBaseWebReader()
documents = loader.load_data(
    root_url="https://support.intercom.com",
    link_selectors=[".article-list a", ".article-list a"],
    article_path="/articles",
    body_selector=".article-body",
    title_selector=".article-title",
    subtitle_selector=".article-subtitle",
)
index = VectorStoreIndex.from_documents(documents)

tools = [
    Tool(
        name="Website Index",
        func=lambda q: index.query(q),
        description=f"Useful when you want answer questions about a product that has a public knowledge base.",
    ),
]
llm = OpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history")
agent_chain = initialize_agent(
    tools, llm, agent="zero-shot-react-description", memory=memory
)

output = agent_chain.run(input="What languages does Intercom support?")
```
