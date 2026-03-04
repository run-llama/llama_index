# Text to Image Tool

This tool allows Agents to use the OpenAI Image endpoint to generate and create variations of images.

## Usage

This tool has more extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-text-to-image/examples/text_to_image.ipynb)

Another example showcases retrieval augmentation over a knowledge corpus with text-to-image. [Notebook](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-text-to-image/examples/text_to_image-pg.ipynb).

```python
from llama_index.tools.text_to_image import TextToImageToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

openai.api_key = "sk-your-key"
tool_spec = TextToImageToolSpec()
# OR
tool_spec = TextToImageToolSpec(api_key="sk-your-key")

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

# Context to store chat history
from llama_index.core.workflow import Context

ctx = Context(agent)


print(
    await agent.run(
        "show 2 images of a beautiful beach with a palm tree at sunset",
        ctx=ctx,
    )
)
print(await agent.run("make the second image higher quality", ctx=ctx))
```

`generate_images`: Generate images from a prompt, specifying the number of images and resolution
`show_images`: Show the images using matplot, useful for Jupyter notebooks
`generate_image_variation`: Generate a variation of an image given a URL.

This loader is designed to be used as a way to load data as a Tool in a Agent.
