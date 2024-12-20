# Amazon Product Extraction Pack

This LlamaPack provides an example of our Amazon product extraction pack.

It loads in a website URL, screenshots the page. Then we use OpenAI GPT-4V + prompt engineering to extract the screenshot into a structured JSON output.

Check out the [notebook here](https://github.com/run-llama/llama-hub/blob/main/llama_hub/llama_packs/amazon_product_extraction/product_extraction.ipynb).

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack AmazonProductExtractionPack --download-dir ./amazon_product_extraction_pack
```

You can then inspect the files at `./amazon_product_extraction_pack` and use them as a template for your own project.

## Code Usage

You can download the pack to a the `./amazon_product_extraction_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
AmazonProductExtractionPack = download_llama_pack(
    "AmazonProductExtractionPack", "./amazon_product_extraction_pack"
)
```

From here, you can use the pack, or inspect and modify the pack in `./amazon_product_extraction_pack`.

Then, you can set up the pack like so:

```python
# create the pack
# get documents from any data loader
amazon_product_extraction_pack = SentenceWindowRetrieverPack(
    amazon_product_page,
)
```

The `run()` function is a light wrapper around `program()`.

```python
response = amazon_product_extraction_pack.run()
display(response.dict())
```

You can also use modules individually.

```python
# get pydantic program
program = amazon_product_extraction_pack.openai_program

# get multi-modal LLM
mm_llm = amazon_product_extraction_pack.openai_mm_llm
```
