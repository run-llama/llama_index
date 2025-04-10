# RAG CLI

One common use case is chatting with an LLM about files you have saved locally on your computer.

We have written a CLI tool to help you do just that! You can point the rag CLI tool to a set of files you've saved locally, and it will ingest those files into a local vector database that is then used for a Chat Q&A repl within your terminal.

By default, this tool uses OpenAI for the embeddings & LLM as well as a local Chroma Vector DB instance. **Warning**: this means that, by default, the local data you ingest with this tool _will_ be sent to OpenAI's API.

However, you do have the ability to customize the models and databases used in this tool. This includes the possibility of running all model execution locally! See the **Customization** section below.

## Setup

To set-up the CLI tool, make sure you've installed the library:

`$ pip install -U llama-index`

You will also need to install [Chroma](../../examples/vector_stores/ChromaIndexDemo.ipynb):

`$ pip install -U chromadb`

After that, you can start using the tool:

```shell
$ llamaindex-cli rag -h
usage: llamaindex-cli rag [-h] [-q QUESTION] [-f FILES [FILES ...]] [-c] [-v] [--clear] [--create-llama]

options:
  -h, --help            show this help message and exit
  -q QUESTION, --question QUESTION
                        The question you want to ask.
  -f, --files FILES [FILES ...]
                        The name of the file(s) or directory you want to ask a question about,such
                        as "file.pdf". Supports globs like "*.py".
  -c, --chat            If flag is present, opens a chat REPL.
  -v, --verbose         Whether to print out verbose information during execution.
  --clear               Clears out all currently embedded data.
  --create-llama        Create a LlamaIndex application based on the selected files.
```

## Usage

Here are some high level steps to get you started:

1. **Set the `OPENAI_API_KEY` environment variable:** By default, this tool uses OpenAI's API. As such, you'll need to ensure the OpenAI API Key is set under the `OPENAI_API_KEY` environment variable whenever you use the tool.
   ```shell
   $ export OPENAI_API_KEY=<api_key>
   ```
1. **Ingest some files:** Now, you need to point the tool at some local files that it can ingest into the local vector database. For this example, we'll ingest the LlamaIndex `README.md` file:
   ```shell
   $ llamaindex-cli rag --files "./README.md"
   ```
   You can also specify a file glob pattern such as:
   ```shell
   $ llamaindex-cli rag --files "./docs/**/*.rst"
   ```
1. **Ask a Question**: You can now start asking questions about any of the documents you'd ingested in the prior step:
   ```shell
   $ llamaindex-cli rag --question "What is LlamaIndex?"
   LlamaIndex is a data framework that helps in ingesting, structuring, and accessing private or domain-specific data for LLM-based applications. It provides tools such as data connectors to ingest data from various sources, data indexes to structure the data, and engines for natural language access to the data. LlamaIndex follows a Retrieval-Augmented Generation (RAG) approach, where it retrieves information from data sources, adds it to the question as context, and then asks the LLM to generate an answer based on the enriched prompt. This approach overcomes the limitations of fine-tuning LLMs and provides a more cost-effective, up-to-date, and trustworthy solution for data augmentation. LlamaIndex is designed for both beginner and advanced users, with a high-level API for easy usage and lower-level APIs for customization and extension.
   ```
1. **Open a Chat REPL**: You can even open a chat interface within your terminal! Just run `$ llamaindex-cli rag --chat` and start asking questions about the files you've ingested.

### Create a LlamaIndex chat application

You can also create a full-stack chat application with a FastAPI backend and NextJS frontend based on the files that you have selected.

To bootstrap the application, make sure you have NodeJS and npx installed on your machine. If not, please refer to the [LlamaIndex.TS](https://ts.llamaindex.ai/docs/llamaindex/getting_started) documentation for instructions.

Once you have everything set up, creating a new application is easy. Simply run the following command:

`$ llamaindex-cli rag --create-llama`

It will call our `create-llama` tool, so you will need to provide several pieces of information to create the app. You can find more information about the `create-llama` on [npmjs - create-llama](https://www.npmjs.com/package/create-llama#example)

```shell
❯ llamaindex-cli rag --create-llama

Calling create-llama using data from /tmp/rag-data/...

✔ What is your project named? … my-app
✔ Which model would you like to use? › gpt-3.5-turbo
✔ Please provide your OpenAI API key (leave blank to skip): …
? How would you like to proceed? › - Use arrow-keys. Return to submit.
   Just generate code (~1 sec)
   Generate code and install dependencies (~2 min)
❯  Generate code, install dependencies, and run the app (~2 min)
...
```

If you choose the option `Generate code, install dependencies, and run the app (~2 min)`, all dependencies will be installed and the app will run automatically. You can then access the application by going to this address: <http://localhost:3000>.

### Supported File Types

Internally, the `rag` CLI tool uses the [SimpleDirectoryReader](../../module_guides/loading/simpledirectoryreader.md) to parse the raw files in your local filesystem into strings.

This module has custom readers for a wide variety of file types. Some of those may require that you `pip install` another module that is needed for parsing that particular file type.

If a file type is encountered with a file extension that the `SimpleDirectoryReader` does not have a custom reader for, it will just read the file as a plain text file.

See the next section for information on how to add your own custom file readers + customize other aspects of the CLI tool!

## Customization

The `rag` CLI tool is highly customizable! The tool is powered by combining the [`IngestionPipeline`](../../module_guides/loading/ingestion_pipeline/index.md) & [`QueryPipeline`](../../module_guides/querying/pipeline/index.md) modules within the [`RagCLI`](https://github.com/run-llama/llama_index/blob/main/llama-index-cli/llama_index/cli/rag/base.py) module.

To create your own custom rag CLI tool, you can simply create a script that instantiates the `RagCLI` class with a `IngestionPipeline` & `QueryPipeline` that you've configured yourself. From there, you can simply run `rag_cli_instance.cli()` in your script to run the same ingestion and Q&A commands against your own choice of embedding models, LLMs, vector DBs, etc.

Here's some high-level code to show the general setup:

```python
#!/path/to/your/virtualenv/bin/python
import os
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.cli.rag import RagCLI


# optional, set any API keys your script may need (perhaps using python-dotenv library instead)
os.environ["OPENAI_API_KEY"] = "sk-xxx"

docstore = SimpleDocumentStore()

vec_store = ...  # your vector store instance
llm = ...  # your LLM instance - optional, will default to OpenAI gpt-3.5-turbo

custom_ingestion_pipeline = IngestionPipeline(
    transformations=[...],
    vector_store=vec_store,
    docstore=docstore,
    cache=IngestionCache(),
)

# Setting up the custom QueryPipeline is optional!
# You can still customize the vector store, LLM, and ingestion transformations without
# having to customize the QueryPipeline
custom_query_pipeline = QueryPipeline()
custom_query_pipeline.add_modules(...)
custom_query_pipeline.add_link(...)

# you can optionally specify your own custom readers to support additional file types.
file_extractor = {".html": ...}

rag_cli_instance = RagCLI(
    ingestion_pipeline=custom_ingestion_pipeline,
    llm=llm,  # optional
    query_pipeline=custom_query_pipeline,  # optional
    file_extractor=file_extractor,  # optional
)

if __name__ == "__main__":
    rag_cli_instance.cli()
```

From there, you're just a few steps away from being able to use your custom CLI script:

1. Make sure to replace the python path at the top to the one your virtual environment is using _(run `$ which python` while your virtual environment is activated)_

1. Let's say you saved your file at `/path/to/your/script/my_rag_cli.py`. From there, you can simply modify your shell's configuration file _(like `.bashrc` or `.zshrc`)_ with a line like `$ export PATH="/path/to/your/script:$PATH"`.
1. After that do `$ chmod +x my_rag_cli.py` to give executable permissions to the file.
1. That's it! You can now just open a new terminal session and run `$ my_rag_cli.py -h`. You can now run the script with the same parameters but using your custom code configurations!
   - Note: you can remove the `.py` file extension from your `my_rag_cli.py` file if you just want to run the command as `$ my_rag_cli --chat`
