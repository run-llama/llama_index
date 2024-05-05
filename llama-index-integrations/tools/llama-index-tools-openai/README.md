# OpenAI Image Generation Tool

This tool allows Agents to generate images using OpenAI's DALL-E model. To see more and get started, visit https://openai.com/blog/dall-e/

## Usage

This tool has a more extensive example usage documented in a Jupyter notebook [here](https://github.com/emptycrown/llama-hub/tree/main/llama_hub/tools/notebooks/openai_image_generation.ipynb)

### Usage with Agent

```python
from llama_index.tools.openai import OpenAIImageGenerationToolSpec

image_generation_tool = OpenAIImageGenerationToolSpec(
    api_key=os.environ["OPENAI_API_KEY"]
)

agent = OpenAIAgent.from_tools(
    [*image_generation_tool.to_tool_list()],
    verbose=True,
)

response = agent.query(
    "A pink and blue llama in a black background with the output"
)

print(response)
```

### Usage directly

```python
from llama_index.tools.openai import OpenAIImageGenerationToolSpec

image_generation_tool = OpenAIImageGenerationToolSpec(
    api_key=os.environ["OPENAI_API_KEY"]
)

image_data = image_generation_tool.image_generation(
    text="A pink and blue llama with a black background",
    response_format="b64_json",
)

image_bytes = base64.b64decode(image_data)

img = Image.open(BytesIO(image_bytes))

display(img)
```

`image_generation`: Takes an text input and generates an image

This loader is designed to be used as a way to load data as a Tool in a Agent.
