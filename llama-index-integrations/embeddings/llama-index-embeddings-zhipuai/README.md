# LlamaIndex Embeddings Integration: ZhipuAI

### Installation

```bash
%pip install llama-index-embeddings-zhipuai
!pip install llama-index
```

### Basic usage

```py
# Import ZhipuAI
from llama_index.embeddings.zhipuai import ZhipuAIEmbedding

embedding = ZhipuAIEmbedding(model="embedding-2", api_key="YOUR API KEY")

response = embedding.get_general_text_embedding("who are you?")
print(response)

# Output: [0.1, 0.2, ...]

import asyncio

response = asyncio.run(embedding.aget_general_text_embedding("who are you?"))
print(response)
# Output: [0.1, 0.2, ...]
```

### ZhipuAI Documentation

https://bigmodel.cn/dev/howuse/introduction
