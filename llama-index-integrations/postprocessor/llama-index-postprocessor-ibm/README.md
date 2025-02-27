# LlamaIndex Postprocessor Integration: IBM

This package integrates the LlamaIndex Postprocessor API with the IBM watsonx.ai Rerank API by leveraging `ibm-watsonx-ai` [SDK](https://ibm.github.io/watsonx-ai-python-sdk/index.html).

## Installation

```bash
pip install llama-index-postprocessor-ibm
```

## Usage

### Setting up

#### Install other required packages:

```bash
pip install -qU llama-index
pip install -qU llama-index-llms-ibm
pip install -qU llama-index-embeddings-ibm
```

To use IBM's Foundation Models, Embeddings and Rerank, you must have an IBM Cloud user API key. Here's how to obtain and set up your API key:

1. **Obtain an API Key:** For more details on how to create and manage an API key, refer to [Managing user API keys](https://cloud.ibm.com/docs/account?topic=account-userapikey&interface=ui).
2. **Set the API Key as an Environment Variable:** For security reasons, it's recommended to not hard-code your API key directly in your scripts. Instead, set it up as an environment variable. You can use the following code to prompt for the API key and set it as an environment variable:

```python
import os
from getpass import getpass

watsonx_api_key = getpass()
os.environ["WATSONX_APIKEY"] = watsonx_api_key
```

Alternatively, you can set the environment variable in your terminal.

- **Linux/macOS:** Open your terminal and execute the following command:

  ```bash
  export WATSONX_APIKEY='your_ibm_api_key'
  ```

  To make this environment variable persistent across terminal sessions, add the above line to your `~/.bashrc`, `~/.bash_profile`, or `~/.zshrc` file.

- **Windows:** For Command Prompt, use:
  ```cmd
  set WATSONX_APIKEY=your_ibm_api_key
  ```

**Note**:

- To provide context for the API call, you must pass the `project_id` or `space_id`. To get your project or space ID, open your project or space, go to the **Manage** tab, and click **General**. For more information see: [Project documentation](https://www.ibm.com/docs/en/watsonx-as-a-service?topic=projects) or [Deployment space documentation](https://www.ibm.com/docs/en/watsonx/saas?topic=spaces-creating-deployment).
- Depending on the region of your provisioned service instance, use one of the urls listed in [watsonx.ai API Authentication](https://ibm.github.io/watsonx-ai-python-sdk/setup_cloud.html#authentication).

In this example, we’ll use the `project_id` and Dallas URL.

Provide `PROJECT_ID` that will be used for initialize each watsonx integration instance.

```python
PROJECT_ID = "PASTE YOUR PROJECT_ID HERE"
URL = "https://us-south.ml.cloud.ibm.com"
```

### Download data and load documents

```bash
!mkdir -p 'data/paul_graham/'
!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'
```

```python
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
```

### Load the Rerank

You might need to adjust rerank parameters for different tasks:

```python
truncate_input_tokens = 512
```

#### Initialize `WatsonxRerank` instance.

You need to specify the `model_id` that will be used for rerank. You can find the list of all the available models in [Supported reranker models](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models-embed.html?context=wx#rerank).

```python
from llama_index.postprocessor.ibm import WatsonxRerank

watsonx_rerank = WatsonxRerank(
    model_id="cross-encoder/ms-marco-minilm-l-12-v2",
    top_n=2,
    url=URL,
    project_id=PROJECT_ID,
    truncate_input_tokens=truncate_input_tokens,
)
```

Alternatively, you can use Cloud Pak for Data credentials. For details, see [watsonx.ai software setup](https://ibm.github.io/watsonx-ai-python-sdk/setup_cpd.html).

```python
from llama_index.postprocessor.ibm import WatsonxRerank

watsonx_rerank = WatsonxRerank(
    model_id="cross-encoder/ms-marco-minilm-l-12-v2",
    url=URL,
    username="PASTE YOUR USERNAME HERE",
    password="PASTE YOUR PASSWORD HERE",
    instance_id="openshift",
    version="5.1",
    project_id=PROJECT_ID,
    truncate_input_tokens=truncate_input_tokens,
)
```

### Load the embedding model

#### Initialize the `WatsonxEmbeddings` instance.

> For more information about `WatsonxEmbeddings` please refer to the `llama-index-embeddings-ibm` package description.

You might need to adjust embedding parameters for different tasks:

```python
truncate_input_tokens = 512
```

You need to specify the `model_id` that will be used for embedding. You can find the list of all the available models in [Supported embedding models](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models-embed.html?context=wx#embed).

```python
from llama_index.embeddings.ibm import WatsonxEmbeddings

watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-30m-english-rtrvr",
    url=URL,
    project_id=PROJECT_ID,
    truncate_input_tokens=truncate_input_tokens,
)
```

Change default settings

```python
from llama_index.core import Settings

Settings.chunk_size = 512
```

#### Build index

```python
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(
    documents=documents, embed_model=watsonx_embedding
)
```

### Load the LLM

#### Initialize the `WatsonxLLM` instance.

> For more information about `WatsonxLLM` please refer to the `llama-index-llms-ibm` package description.

You need to specify the `model_id` that will be used for inferencing. You can find the list of all the available models in [Supported foundation models](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models.html?context=wx).

You might need to adjust model `parameters` for different models or tasks. For details, refer to [Available MetaNames](https://ibm.github.io/watsonx-ai-python-sdk/fm_model.html#metanames.GenTextParamsMetaNames).

```python
max_new_tokens = 128
```

```python
from llama_index.llms.ibm import WatsonxLLM

watsonx_llm = WatsonxLLM(
    model_id="meta-llama/llama-3-3-70b-instruct",
    url=URL,
    project_id=PROJECT_ID,
    max_new_tokens=max_new_tokens,
)
```

### Send a query

#### Retrieve top 10 most relevant nodes, then filter with `WatsonxRerank`

```python
query_engine = index.as_query_engine(
    llm=watsonx_llm,
    similarity_top_k=10,
    node_postprocessors=[watsonx_rerank],
)
response = query_engine.query(
    "What did Sam Altman do in this essay?",
)
```

```python
from llama_index.core.response.pprint_utils import pprint_response

pprint_response(response, show_source=True)
```

#### Directly retrieve top 2 most similar nodes

```python
query_engine = index.as_query_engine(
    llm=watsonx_llm,
    similarity_top_k=2,
)
response = query_engine.query(
    "What did Sam Altman do in this essay?",
)
```

Retrieved context is irrelevant and response is hallucinated.

```python
pprint_response(response, show_source=True)
```
