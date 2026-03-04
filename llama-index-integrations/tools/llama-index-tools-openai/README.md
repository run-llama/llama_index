# OpenAI Image Generation Tool

This tool allows Agents to generate images using OpenAI's DALL-E model. To see more and get started, visit https://openai.com/blog/dall-e/

## Usage

This tool has a more extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-openai/examples/multimodal_openai_image.ipynb).

### Usage with Agent

```python
from llama_index.tools.openai import OpenAIImageGenerationToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

image_generation_tool = OpenAIImageGenerationToolSpec(
    api_key=os.environ["OPENAI_API_KEY"]
)

agent = FunctionAgent(
    tools=[*image_generation_tool.to_tool_list()],
    llm=OpenAI(model="gpt-4.1"),
)

print(
    await agent.run(
        "A pink and blue llama in a black background with the output"
    )
)
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
