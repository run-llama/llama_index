# Beautiful Soup Website Loader

```bash
pip install llama-index-readers-web
```

This loader is a web scraper that fetches the text from websites using the `Beautiful Soup` (aka `bs4`) Python package. Furthermore, the flexibility of Beautiful Soup allows for custom templates that enable the loader to extract the desired text from specific website designs, such as Substack. Check out the code to see how to add your own.

## Usage

To use this loader, you need to pass in an array of URLs.

```python
from llama_index.readers.web import BeautifulSoupWebReader

loader = BeautifulSoupWebReader()
documents = loader.load_data(urls=["https://google.com"])
```

You can also add your own specific website parsers in `base.py` that automatically get used for certain URLs. Alternatively, you may tell the loader to use a certain parser by passing in the `custom_hostname` argument. For reference, this is what the Beautiful Soup parser looks like for Substack sites:

```python
def _substack_reader(soup: Any) -> Tuple[str, Dict[str, Any]]:
    """Extract text from Substack blog post."""
    extra_info = {
        "Title of this Substack post": soup.select_one(
            "h1.post-title"
        ).getText(),
        "Subtitle": soup.select_one("h3.subtitle").getText(),
        "Author": soup.select_one("span.byline-names").getText(),
    }
    text = soup.select_one("div.available-content").getText()
    return text, extra_info
```

## Examples

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).

### LlamaIndex

```python
from llama_index.core import VectorStoreIndex, download_loader

from llama_index.readers.web import BeautifulSoupWebReader

loader = BeautifulSoupWebReader()
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

from llama_index.readers.web import BeautifulSoupWebReader

loader = BeautifulSoupWebReader()
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

## Custom hostname example

To use a custom hostname like readme.co, substack.com or any other commonly-used website template, you can pass in the `custom_hostname` argument to guarantee that a custom parser is used (if it exists). Check out the code to see which ones are currently implemented.

```python
documents = loader.load_data(
    urls=["https://langchain.readthedocs.io/en/latest/"],
    custom_hostname="readthedocs.io",
)
```
