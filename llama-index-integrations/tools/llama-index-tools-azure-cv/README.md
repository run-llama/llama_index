# Azure Computer Vision Tool

This tool connects to a Azure account and allows an Agent to perform a variety of computer vision tasks on image urls.

You will need to set up an api key and computer vision instance using Azure, learn more here: https://azure.microsoft.com/en-ca/products/cognitive-services/computer-vision

## Usage

This tool has a more extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-azure-cv/examples/azure_vision.ipynb)

Here's an example usage of the AzureCVToolSpec.

```python
from llama_index.tools.azure_cv import AzureCVToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

tool_spec = AzureCVToolSpec(api_key="your-key", resource="your-resource")

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(), llm=OpenAI(model="gpt-4.1")
)

await agent.run(
    "caption this image and tell me what tags are in it https://portal.vision.cognitive.azure.com/dist/assets/ImageCaptioningSample1-bbe41ac5.png"
)
await agent.run(
    "caption this image and read any text https://portal.vision.cognitive.azure.com/dist/assets/OCR3-4782f088.jpg"
)
```

`process_image`: Send an image for computer vision classification of objects, tags, captioning or OCR.

This loader is designed to be used as a way to load data as a Tool in a Agent.
