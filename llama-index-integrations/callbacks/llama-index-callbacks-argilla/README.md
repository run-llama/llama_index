<div align="center">
  <h1>✨🦙 Argilla's LlamaIndex Integration</h1>
  <p><em> Argilla integration into the LlamaIndex workflow</em></p>
</div>

> [!TIP]
> To discuss, get support, or give feedback [join Argilla's Slack Community](https://join.slack.com/t/rubrixworkspace/shared_invite/zt-whigkyjn-a3IUJLD7gDbTZ0rKlvcJ5g) and you will be able to engage with our amazing community and also with the core developers of `argilla` and `distilabel`.

This integration allows the user to include the feedback loop that Argilla offers into the LlamaIndex ecosystem. It's based on a callback handler to be run within the LlamaIndex workflow.

Don't hesitate to check out both [LlamaIndex](https://github.com/run-llama/llama_index) and [Argilla](https://github.com/argilla-io/argilla)

## Getting Started

You first need to install argilla and argilla-llama-index as follows:

```bash
pip install llama-index-callbacks-argilla
```

You will need to an Argilla Server running to monitor the LLM. You can either install the server locally or have it on HuggingFace Spaces. For a complete guide on how to install and initialize the server, you can refer to the [Quickstart Guide](https://docs.argilla.io/en/latest/getting_started/quickstart_installation.html).

## Usage

It requires just a simple step to log your data into Argilla within your LlamaIndex workflow. We just need to call the handler before starting production with your LLM.

We will use GPT3.5 from OpenAI as our LLM. For this, you will need a valid API key from OpenAI. You can have more info and get one via [this link](https://openai.com/blog/openai-api).

After you get your API key, the easiest way to import it is through an environment variable, or via _getpass()_.

```python
import os
from getpass import getpass

openai_api_key = os.getenv("OPENAI_API_KEY", None) or getpass(
    "Enter OpenAI API key:"
)
```

Let's now write all the necessary imports

```python
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    set_global_handler,
)
from llama_index.llms.openai import OpenAI
```

What we need to do is to set Argilla as the global handler as below. Within the handler, we need to provide the dataset name that we will use. If the dataset does not exist, it will be created with the given name. You can also set the API KEY, API URL, and the Workspace name. You can learn more about the variables that controls Argilla initialization [here](https://docs.argilla.io/en/latest/getting_started/installation/configurations/workspace_management.html)

> [!TIP]
> Remember that the default Argilla workspace name is `admin`. If you want to use a custom Workspace, you'll need to create it and grant access to the desired users. The link above also explains how to do that.

```python
set_global_handler("argilla", dataset_name="query_model")
```

Let's now create the llm instance, using GPT-3.5 from OpenAI.

```python
llm = OpenAI(
    model="gpt-3.5-turbo", temperature=0.8, openai_api_key=openai_api_key
)
```

With the code snippet below, you can create a basic workflow with LlamaIndex. You will also need a txt file as the data source within a folder named "data". For a sample data file and more info regarding the use of Llama Index, you can refer to the [Llama Index documentation](https://docs.llamaindex.ai/en/stable/getting_started/starter_example.html).

```python
docs = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine()
```

Now, let's run the `query_engine` to have a response from the model.

```python
response = query_engine.query("What did the author do growing up?")
response
```

```bash
The author worked on two main things outside of school before college: writing and programming. They wrote short stories and tried writing programs on an IBM 1401. They later got a microcomputer, built it themselves, and started programming on it.
```

The prompt given and the response obtained will be logged in to Argilla server.
