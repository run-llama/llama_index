# Starter Tutorial (Local Models)

!!! tip
    Make sure you've followed the [custom installation](installation.md) steps first.

This is our famous "5 lines of code" starter example with local LLM and embedding models. We will use `nomic-embed-text` as our embedding model and `Llama3`, both served through `Ollama`.

## Download data

This example uses the text of Paul Graham's essay, ["What I Worked On"](http://paulgraham.com/worked.html). This and many other examples can be found in the `examples` folder of our repo.

The easiest way to get it is to [download it via this link](https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt) and save it in a folder called `data`.

## Setup

Ollama is a tool to help you get set up with LLMs locally (currently supported on OSX and Linux. You can install Ollama on Windows through WSL 2).

Follow the [README](https://github.com/jmorganca/ollama) to learn how to install it.

To download the Llama3 model just do `ollama pull llama3`.

To download the nomic embeddings, just do `ollama pull nomic-embed-text`

**NOTE**: You will need a machine with at least 32GB of RAM.

To import `llama_index.llms.ollama`, you should run `pip install llama-index-llms-ollama`.

To import `llama_index.embeddings.ollama`, you should run `pip install llama-index-embeddings-ollama`.

More integrations are all listed on https://llamahub.ai.

## Load data and build an index

In the same folder where you created the `data` folder, create a file called `starter.py` file with the following:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

documents = SimpleDirectoryReader("data").load_data()

# nomic embedding model
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# ollama
Settings.llm = Ollama(model="llama3", request_timeout=360.0)

index = VectorStoreIndex.from_documents(
    documents,
)
```

This builds an index over the documents in the `data` folder (which in this case just consists of the essay text, but could contain many documents).

Your directory structure should look like this:

<pre>
├── starter.py
└── data
    └── paul_graham_essay.txt
</pre>

We use the `nomic-embed-text` from our `Ollama` embedding wrapper. We also use our `Ollama` LLM wrapper to load in the Llama3 model.

## Query your data

Add the following lines to `starter.py`

```python
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)
```

This creates an engine for Q&A over your index and asks a simple question. You should get back a response similar to the following: `The author wrote short stories and tried to program on an IBM 1401.`

You can view logs, persist/load the index similar to our [starter example](starter_example.md).

!!! tip
    - learn more about the [high-level concepts](./concepts.md).
    - tell me how to [customize things](./customization.md).
    - curious about a specific module? check out the [component guides](../module_guides/index.md).
