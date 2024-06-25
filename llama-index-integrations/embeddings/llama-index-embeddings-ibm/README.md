# LlamaIndex Embeddings Integration: IBM

This package provides the integration between LlamaIndex and IBM watsonx.ai through the `ibm-watsonx-ai` SDK.

## Installation

```bash
pip install llama-index-embeddings-ibm
```

## Usage

### Setting up

To use IBM's models, you must have an IBM Cloud user API key. Here's how to obtain and set up your API key:

1. **Obtain an API Key:** For more details on how to create and manage an API key, refer to IBM's [documentation](https://cloud.ibm.com/docs/account?topic=account-userapikey&interface=ui).
2. **Set the API Key as an Environment Variable:** For security reasons, it's recommended to not hard-code your API key directly in your scripts. Instead, set it up as an environment variable. You can use the following code to prompt for the API key and set it as an environment variable:

```python
import os
from getpass import getpass

watsonx_api_key = getpass()
os.environ["WATSONX_APIKEY"] = watsonx_api_key
```

In alternative, you can set the environment variable in your terminal.

- **Linux/macOS:** Open your terminal and execute the following command:

  ```bash
  export WATSONX_APIKEY='your_ibm_api_key'
  ```

  To make this environment variable persistent across terminal sessions, add the above line to your `~/.bashrc`, `~/.bash_profile`, or `~/.zshrc` file.

- **Windows:** For Command Prompt, use:
  ```cmd
  set WATSONX_APIKEY=your_ibm_api_key
  ```

### Load the model

You might need to adjust embedding parameters for different tasks.

```python
truncate_input_tokens = 3
```

Initialize the `WatsonxEmbeddings` class with previously set parameters.

**Note**:

- To provide context for the API call, you must add `project_id` or `space_id`. For more information see [documentation](https://www.ibm.com/docs/en/watsonx-as-a-service?topic=projects).
- Depending on the region of your provisioned service instance, use one of the urls described [here](https://ibm.github.io/watsonx-ai-python-sdk/setup_cloud.html#authentication).

In this example, we’ll use the `project_id` and Dallas url.

You need to specify `model_id` that will be used for inferencing.

```python
from llama_index.embeddings.ibm import WatsonxEmbeddings

watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="PASTE YOUR PROJECT_ID HERE",
    truncate_input_tokens=truncate_input_tokens,
)
```

Alternatively you can use Cloud Pak for Data credentials. For details, see [documentation](https://ibm.github.io/watsonx-ai-python-sdk/setup_cpd.html).

```python
watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="PASTE YOUR URL HERE",
    username="PASTE YOUR USERNAME HERE",
    password="PASTE YOUR PASSWORD HERE",
    instance_id="openshift",
    version="5.0",
    project_id="PASTE YOUR PROJECT_ID HERE",
    truncate_input_tokens=truncate_input_tokens,
)
```

## Usage

### Embed query

```python
query = "Example query."

query_result = watsonx_embedding.get_query_embedding(query)
print(query_result[:5])
```

### Embed list of texts

```python
texts = ["This is a content of one document", "This is another document"]

doc_result = watsonx_embedding.get_text_embedding_batch(texts)
print(doc_result[0][:5])
```
