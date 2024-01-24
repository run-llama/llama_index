# RAG CLI

One common use case is chatting with an LLM about files you have saved locally on your machine.

We have written a CLI tool do help you do just that very easily! You can point the rag CLI tool to a set of files you've saved locally, and it will ingest those files into a local vector database that is then used for a Chat Q&A repl within your terminal.

By default, this tool uses OpenAI for the embeddings & LLM as well as a local Chroma Vector DB instance. **Warning**: this means the local data you ingest with this tool _will_ be sent to OpenAI's API.

## Setup

To set-up the CLI tool, make sure you've installed the library: `pip install llama-index`.

You will also need to install [Chroma](/docs/examples/vector_stores/ChromaIndexDemo.ipynb): `pip install chromadb`

After that, you can start using the tool:

```
$ llamaindex-cli rag -h
usage: llamaindex-cli rag [-h] [-q QUESTION] [-f FILES] [-c] [-v] [--clear]

options:
  -h, --help            show this help message and exit
  -q QUESTION, --question QUESTION
                        The question you want to ask.
  -f FILES, --files FILES
                        The name of the file or directory you want to ask a question about,such as "file.pdf".
  -c, --chat            If flag is present, opens a chat REPL.
  -v, --verbose         Whether to print out verbose information during execution.
  --clear               Clears out all currently embedded data.
```

## Usage

Here are some high level steps to get you started:

1. **Set the `OPENAI_API_KEY` environment variable:** By default, this tool uses OpenAI's API. As such, you'll need to ensure the OpenAI API Key is set under the `OPENAI_API_KEY` environment variable whenever you use the tool.
   ```
   export OPENAI_API_KEY=<api_key>
   ```
1. **Ingest some files:** Now, you need to point the tool at some local files that it can ingest into the local vector database. For this example, we'll ingest the LlamaIndex `README.md` file:
   ```
   $ llamaindex-cli rag --files "./README.md"
   ```
   You can also specify a file glob pattern such as:
   ```
   llamaindex-cli rag --files "./docs/**/*.rst"
   ```
1. **Ask a Question**: You can now start asking questions about any of the documents you'd ingested in the prior step:
   ```
   $ llamaindex-cli rag --question "What is LlamaIndex?"
   LlamaIndex is a data framework that helps in ingesting, structuring, and accessing private or domain-specific data for LLM-based applications. It provides tools such as data connectors to ingest data from various sources, data indexes to structure the data, and engines for natural language access to the data. LlamaIndex follows a Retrieval-Augmented Generation (RAG) approach, where it retrieves information from data sources, adds it to the question as context, and then asks the LLM to generate an answer based on the enriched prompt. This approach overcomes the limitations of fine-tuning LLMs and provides a more cost-effective, up-to-date, and trustworthy solution for data augmentation. LlamaIndex is designed for both beginner and advanced users, with a high-level API for easy usage and lower-level APIs for customization and extension.
   ```
1. **Open a Chat REPL**: You can even open a chat interface within your terminal! Just run `llamaindex-cli rag --chat` and start asking questions about the files you've ingested.

## Customization

The `rag` CLI tool is highly customizable! The tool is powered by stitching together the [`IngestionPipeline`](https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/root.html) & [`QueryPipeline`](https://docs.llamaindex.ai/en/stable/module_guides/querying/pipeline/root.html) modules within the [`RagCLI`](https://github.com/run-llama/llama_index/blob/main/llama_index/command_line/rag.py) module. To create your own custom rag CLI tool, you can simply create an script that instantiates the `RagCLI` class with a `IngestionPipeline` & `QueryPipeline` that you've configured yourself. From there, you can simply run `rag_cli_instance.cli()` in your script to run the same ingestion and Q&A commands against your own choice of embedding models, LLMs, vector DBs, etc.
