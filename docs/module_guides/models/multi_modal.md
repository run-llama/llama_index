# [Beta] Multi-modal models

## Concept

Large language models (LLMs) are text-in, text-out. Large Multi-modal Models (LMMs) generalize this beyond the text modalities. For instance, models such as GPT-4V allow you to jointly input both images and text, and output text.

We've included a base `MultiModalLLM` abstraction to allow for text+image models. **NOTE**: This naming is subject to change!

## Usage Pattern

The following code snippet shows how you can get started using LMMs e.g. with GPT-4V.

```python
from llama_index.multi_modal_llms import OpenAIMultiModal
from llama_index.multi_modal_llms.generic_utils import (
    load_image_urls,
)
from llama_index import SimpleDirectoryReader

# load image documents from urls
image_documents = load_image_urls(image_urls)

# load image documents from local directory
image_documents = SimpleDirectoryReader(local_directory).load_data()

# non-streaming
openai_mm_llm = OpenAIMultiModal(
    model="gpt-4-vision-preview", api_key=OPENAI_API_TOKEN, max_new_tokens=300
)
response = openai_mm_llm.complete(
    prompt="what is in the image?", image_documents=image_documents
)
```

**Legend**

- ✅ = should work fine
- ⚠️ = sometimes unreliable, may need to improve
- 🛑 = not available at the moment. Support on the way

### End to End Multi-Modal Work Flow

| Query Type | Data Sources<br>for MultiModal<br>Vector Store/Index | MultiModal<br>Embedding                | Retriever                                        | Query<br>Engine        | Output<br>Data<br>Type                   |
| ---------- | ---------------------------------------------------- | -------------------------------------- | ------------------------------------------------ | ---------------------- | ---------------------------------------- |
| Text ✅    | Text ✅                                              | Text ✅                                | Top-k retrieval ✅<br>Simple Fusion retrieval ✅ | Simple Query Engine ✅ | Retrieved Text ✅<br>Generated Text ✅   |
| Image ✅   | Image ✅                                             | Image ✅<br>Image to Text Embedding ✅ | Top-k retrieval ✅<br>Simple Fusion retrieval ✅ | Simple Query Engine ✅ | Retrieved Image ✅<br>Generated Image 🛑 |
| Audio 🛑   | Audio 🛑                                             | Audio 🛑                               | 🛑                                               | 🛑                     | Audio 🛑                                 |
| Video 🛑   | Video 🛑                                             | Video 🛑                               | 🛑                                               | 🛑                     | Video 🛑                                 |

### Multi-Modal LLM Models

Outer pipes Cell padding
No sorting
| Multi-Modal<br>Vision Models | Single<br>Image<br>Reasoning | Multiple<br>Images<br>Reasoning | Image<br>Embeddings | Simple<br>Query<br>Engine |
| ----------------------------------------------------------- | ---------------------------- | ------------------------------- | ------------------- | ------------------------- |
| GPT4V<br>(OpenAI API) | ✅ | ✅ | 🛑 | ✅ |
| CLIP<br>(Local host) | 🛑 | 🛑 | ✅ | 🛑 |
| LLaVa<br>(replicate) | ✅ | 🛑 | 🛑 | ✅ |
| Fuyu-8B<br>(replicate) | ✅ | 🛑 | 🛑 | ✅ |
| ImageBind<br>[To integrate] | 🛑 | 🛑 | ✅ | 🛑 |
| [MiniGPT-4<br>](https://minigpt-4.github.io/)[To integrate] | ✅ | 🛑 | 🛑 | ✅ |

### Multi Modal Vector Stores

| Multi-Modal<br>Vector Stores                                                                                              | Single<br>Vector<br>Store | Multiple<br>Vector<br>Store | Text<br>Embedding                                         | Image<br>Embedding                                      |
| ------------------------------------------------------------------------------------------------------------------------- | ------------------------- | --------------------------- | --------------------------------------------------------- | ------------------------------------------------------- |
| LLamaIndex self-built<br>MultiModal Index                                                                                 | 🛑                        | ✅                          | Can be arbitrary<br>text embedding<br>(Default is GPT3.5) | Can be arbitrary<br>text embedding<br>(Default is CLIP) |
| Chroma                                                                                                                    | ✅                        | 🛑                          | CLIP ✅                                                   | CLIP ✅                                                 |
| [Weaviate](https://weaviate.io/developers/weaviate/modules/retriever-vectorizer-modules/multi2vec-bind)<br>[To integrate] | ✅                        | 🛑                          | CLIP ✅<br>ImageBind ✅                                   | CLIP ✅<br>ImageBind ✅                                 |

## Modules

We support integrations with GPT-4V, LLaVA, and more.

```{toctree}
---
maxdepth: 1
---
/examples/multi_modal/openai_multi_modal.ipynb
/examples/multi_modal/replicate_multi_modal.ipynb
/examples/multi_modal/multi_modal_retrieval.ipynb
/examples/multi_modal/llava_multi_modal_tesla_10q.ipynb
```
