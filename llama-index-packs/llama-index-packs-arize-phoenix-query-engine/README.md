# Arize-Phoenix LlamaPack

This LlamaPack instruments your LlamaIndex app for LLM tracing with [Phoenix](https://github.com/Arize-ai/phoenix), an open-source LLM observability library from [Arize AI](https://phoenix.arize.com/).

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack ArizePhoenixQueryEnginePack --download-dir ./arize_pack
```

You can then inspect the files at `./arize_pack` and use them as a template for your own project!

## Code Usage

You can download the pack to a the `./arize_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
ArizePhoenixQueryEnginePack = download_llama_pack(
    "ArizePhoenixQueryEnginePack", "./arize_pack"
)
```

You can then inspect the files at `./arize_pack` or continue on to use the module.

```python
import os

from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.web import SimpleWebPageReader
from tqdm.auto import tqdm
```

Configure your OpenAI API key.

```python
os.environ["OPENAI_API_KEY"] = "copy-your-openai-api-key-here"
```

Parse your documents into a list of nodes and pass to your LlamaPack. In this example, use nodes from a Paul Graham essay as input.

```python
documents = SimpleWebPageReader().load_data(
    [
        "https://raw.githubusercontent.com/jerryjliu/llama_index/adb054429f642cc7bbfcb66d4c232e072325eeab/examples/paul_graham_essay/data/paul_graham_essay.txt"
    ]
)
parser = SentenceSplitter()
nodes = parser.get_nodes_from_documents(documents)
phoenix_pack = ArizePhoenixQueryEnginePack(nodes=nodes)
```

Run a set of queries via the pack's `run` method, which delegates to the underlying query engine.

```python
queries = [
    "What did Paul Graham do growing up?",
    "When and how did Paul Graham's mother die?",
    "What, in Paul Graham's opinion, is the most distinctive thing about YC?",
    "When and how did Paul Graham meet Jessica Livingston?",
    "What is Bel, and when and where was it written?",
]
for query in tqdm(queries):
    print("Query")
    print("=====")
    print(query)
    print()
    response = phoenix_pack.run(query)
    print("Response")
    print("========")
    print(response)
    print()
```

View your trace data in the Phoenix UI.

```python
phoenix_session_url = phoenix_pack.get_modules()["session_url"]
print(f"Open the Phoenix UI to view your trace data: {phoenix_session_url}")
```

You can access the internals of the LlamaPack, including your Phoenix session and your query engine, via the `get_modules` method.

```python
phoenix_pack.get_modules()
```

Check out the [Phoenix documentation](https://docs.arize.com/phoenix/) for more information!
