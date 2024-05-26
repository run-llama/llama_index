# LlamaIndex Llms Integration: IBM

This package provides the integration between LlamaIndex and IBM watsonx.ai through the `ibm-watsonx-ai` SDK.

## Installation

```bash
pip install llama-index-llms-ibm
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

### Loading the model

You might need to adjust model parameters for different models or tasks. For more details on the parameters, refer to IBM's [documentation](https://ibm.github.io/watsonx-ai-python-sdk/fm_model.html#metanames.GenTextParamsMetaNames).

```python
temperature = 0.5
max_new_tokens = 50
additional_params = {
    "min_new_tokens": 1,
    "top_k": 50,
}
```

Initialize the WatsonxLLM class with the previously set parameters.

```python
from llama_index.llms.ibm import WatsonxLLM

watsonx_llm = WatsonxLLM(
    model_id="PASTE THE CHOSEN MODEL_ID HERE",
    url="PASTE YOUR URL HERE",
    project_id="PASTE YOUR PROJECT_ID HERE",
    temperature=temperature,
    max_new_tokens=max_new_tokens,
    additional_params=additional_params,
)
```

**Note:**

- You must provide a `project_id` or `space_id`. For more information refer to IBM's [documentation](https://www.ibm.com/docs/en/watsonx-as-a-service?topic=projects).
- Depending on the region of your provisioned service instance, use one of the urls described [here](https://ibm.github.io/watsonx-ai-python-sdk/setup_cloud.html#authentication).
- You need to specify the model you want to use for inferencing through `model_id`. You can find the list of available models [here](https://ibm.github.io/watsonx-ai-python-sdk/fm_model.html#ibm_watsonx_ai.foundation_models.utils.enums.ModelTypes).

Alternatively you can use Cloud Pak for Data credentials. For more details, refer to IBM's [documentation](https://ibm.github.io/watsonx-ai-python-sdk/setup_cpd.html).

```python
watsonx_llm = WatsonxLLM(
    model_id="ibm/granite-13b-instruct-v2",
    url="PASTE YOUR URL HERE",
    username="PASTE YOUR USERNAME HERE",
    password="PASTE YOUR PASSWORD HERE",
    instance_id="openshift",
    version="4.8",
    project_id="PASTE YOUR PROJECT_ID HERE",
    temperature=temperature,
    max_new_tokens=max_new_tokens,
    additional_params=additional_params,
)
```

### Create a Completion

Below is an example that shows how to call the model directly using a string type prompt

```python
response = watsonx_llm.complete("What is a Generative AI?")
print(response)
```

### Calling `chat` with a list of messages

To create `chat` completions by providing a list of messages, use following

```python
from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(role="system", content="You are an AI assistant"),
    ChatMessage(role="user", content="Who are you?"),
]
response = watsonx_llm.chat(messages)
print(response)
```

### Streaming the Model output

You can stream the model output.

```python
for chunk in watsonx_llm.stream_complete(
    "Describe your favorite city and why it is your favorite."
):
    print(chunk.delta, end="")
```
