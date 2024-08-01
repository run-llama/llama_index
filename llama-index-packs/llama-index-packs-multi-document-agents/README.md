# Multi-Document Agents Pack

This LlamaPack provides an example of our multi-document agents.

This specific template shows the e2e process of building this. Given a set
of documents, the pack will build our multi-document agents architecture.

- setup a document agent over agent doc (capable of QA and summarization)
- setup a top-level agent over doc agents
- During query-time, do "tool retrieval" to return the set of relevant candidate documents, and then do retrieval within each document.

Check out the [notebook here](https://github.com/run-llama/llama-hub/blob/main/llama_hub/llama_packs/multi_document_agents/multi_document_agents.ipynb).

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack MultiDocumentAgentsPack --download-dir ./multi_doc_agents_pack
```

You can then inspect the files at `./multi_doc_agents_pack` and use them as a template for your own project.

## Code Usage

You can download the pack to a the `./multi_doc_agents_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
MultiDocumentAgentsPack = download_llama_pack(
    "MultiDocumentAgentsPack", "./multi_doc_agents_pack"
)
```

From here, you can use the pack, or inspect and modify the pack in `./multi_doc_agents_pack`.

Then, you can set up the pack like so:

```python
# imagine documents on different cities
docs = ...

doc_titles = ["Toronto", "Seattle", "Houston", "Chicago"]
doc_descriptions = [
    "<Toronto description>",
    "<Seattle description>",
    "<Houston description>",
    "<Chicago description>",
]

# create the pack
# get documents from any data loader
multi_doc_agents_pack = MultiDocumentAgentsPack(
    docs, doc_titles, doc_descriptions
)
```

The `run()` function is a light wrapper around `query_engine.query()`.

```python
response = multi_doc_agents_pack.run(
    "Tell me the demographics of Houston, and then compare with the demographics of Chicago"
)
```

You can also use modules individually.

```python
# get the top-level agent
top_agent = multi_doc_agents_pack.top_agent

# get the object index (which indexes all document agents, can return top-k
# most relevant document agents as tools given user query)
obj_index = multi_doc_agents_pack.obj_index

# get document agents
doc_agents = multi_doc_agents_pack.agents
```
