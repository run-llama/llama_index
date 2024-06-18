# LlamaIndex Llms Integration: Watsonx

The usage of `llama-index-llms-watsonx` is deprecated in favor of `llama-index-llms-ibm`. To install recommended package that uses `ibm-watsonx-ai` underneath, run `pip install -qU llama-index-llms-ibm`. Use following to load the model using an IBM watsonx.ai integration

```python
from llama_index.llms.ibm import WatsonxLLM

watsonx_llm = WatsonxLLM(
    model_id="PASTE THE CHOSEN MODEL_ID HERE",
    url="PASTE YOUR URL HERE",
    apikey="PASTE YOUR IBM APIKEY HERE",
    project_id="PASTE YOUR PROJECT_ID HERE",
    temperature=temperature,
    max_new_tokens=max_new_tokens,
    additional_params=additional_params,
)
```
