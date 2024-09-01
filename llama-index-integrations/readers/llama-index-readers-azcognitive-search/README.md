# Azure Cognitive Search Loader

```bash
pip install llama-index-readers-azcognitive-search
```

The AzCognitiveSearchReader Loader returns a set of texts corresponding to documents retrieved from specific index of Azure Cognitive Search.
The user initializes the loader with credentials (service name and key) and the index name.

## Usage

Here's an example usage of the AzCognitiveSearchReader.

```python
from llama_index.readers.azcognitive_search import AzCognitiveSearchReader

reader = AzCognitiveSearchReader(
    "<Azure_Cognitive_Search_NAME>",
    "<Azure_Cognitive_Search_KEY>",
    "<Index_name>",
)


query_sample = ""
documents = reader.load_data(
    query="<search_term>",
    content_field="<content_field_name>",
    filter="<azure_search_filter>",
)
```

## Usage in combination with langchain

```python
from llama_index.core import VectorStoreIndex, download_loader
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import Tool, AgentExecutor, load_tools, initialize_agent

from llama_index.readers.azcognitive_search import AzCognitiveSearchReader

az_loader = AzCognitiveSearchReader(
    COGNITIVE_SEARCH_SERVICE_NAME, COGNITIVE_SEARCH_KEY, INDEX_NAME
)

documents = az_loader.load_data(query, field_name)

index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)

tools = [
    Tool(
        name="Azure cognitive search index",
        func=lambda q: index.query(q),
        description=f"Useful when you want answer questions about the text on azure cognitive search.",
    ),
]
memory = ConversationBufferMemory(memory_key="chat_history")
agent_chain = initialize_agent(
    tools, llm, agent="zero-shot-react-description", memory=memory
)

result = agent_chain.run(input="How can I contact with my health insurance?")
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
