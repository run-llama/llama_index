# LLaVA Completion Pack

This LlamaPack creates the LLaVA multimodal model, and runs its `complete` endpoint to execute queries.

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack LlavaCompletionPack --download-dir ./llava_pack
```

You can then inspect the files at `./llava_pack` and use them as a template for your own project!

## Code Usage

You can download the pack to a `./llava_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
LlavaCompletionPack = download_llama_pack(
    "LlavaCompletionPack", "./llava_pack"
)
```

From here, you can use the pack, or inspect and modify the pack in `./llava_pack`.

Then, you can set up the pack like so:

```python
# create the pack
llava_pack = LlavaCompletionPack(image_url="./images/image1.jpg")
```

The `run()` function is a light wrapper around `llm.complete()`.

```python
response = llava_pack.run(
    "What dinner can I cook based on the picture of the food in the fridge?"
)
```

You can also use modules individually.

```python
# call the llm.complete()
llm = llava_pack.llm
response = llm.complete("query_str")
```
