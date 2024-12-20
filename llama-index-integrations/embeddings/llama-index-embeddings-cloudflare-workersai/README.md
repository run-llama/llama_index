# LlamaIndex Embeddings Integration: Cloudflare Workers AI

Cloudflare Workers AI provides text embedding service for Cloudflare users. You can find a full list of available models from its document
https://developers.cloudflare.com/workers-ai/models/#text-embeddings

To learn more about Cloudflare Workers AI in general, visit https://developers.cloudflare.com/workers-ai/

## Example

```shell
pip install llama-index-embeddings-cloudflare-workersai
```

```py
from llama_index.embeddings.cloudflare_workersai import CloudflareEmbedding

my_account_id = "example_account_id"
my_api_token = "example_token"

my_embed = CloudflareEmbedding(
    account_id=my_account_id,
    auth_token=my_api_token,
    model="@cf/baai/bge-base-en-v1.5",
)

embeddings = my_embed.get_text_embedding("Why sky is blue")

embeddings_batch = my_embed.get_text_embedding_batch(
    ["Why sky is blue", "Why roses are red"]
)
```

For more detailed example of installation and usage, please refer to the [Jupyter Notebook example](https://docs.llamaindex.ai/en/stable/examples/embeddings/cloudflare_workersai/).
