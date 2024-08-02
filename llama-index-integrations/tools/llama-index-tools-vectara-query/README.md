## Vectara Query Tool

This tool connects to a Vectara corpus and allows agents to make retrieval augmented generation (RAG) queries to retrieve search results and summarized responses based on documents in a corpus.

## Usage

This tool has a more extensive example usage documented in a Jupyter notebok [here](valid_link_to_notebok)

To use this tool, you'll need the following information in your environment:

- `VECTARA_CUSTOMER_ID`: The customer id for your Vectara account. If you don't have an account, you can create one [here](https://console.vectara.com/signup).
- `VECTARA_CORPUS_ID`: The corpus id for the Vectara corpus that you want your tool to search for information.
- `VECTARA_API_KEY`: An API key that can perform queries on this corpus.

Here's an example usage of the VectaraQueryToolSpec.

```python
from llama_index.tools.vectara_query import VectaraQueryToolSpec
from llama_index.agent.openai import OpenAIAgent

tool_spec = VectaraQueryToolSpec(
    vectara_customer_id=os.environ["VECTARA_CUSTOMER_ID"],
    vectara_corpus_id=os.environ["VECTARA_CORPUS_ID"],
    vectara_api_key=os.environ["VECTARA_API_KEY"],
)

agent = OpenAIAgent.from_tools(tool_spec.to_tool_list())

agent.chat("WHATEVER GENERAL EXAMPLE QUERY WE WANT HERE")
```

The available tools are:

`semantic_search`: A tool that accepts a query and other parameters and uses RAG to obtain the top search results.

`rag_query`: A tool that accepts a query and other parameters and uses RAG to obtain the top search results and generate a summary of the retrieved documents.
