# [Beta] Multi-modal models

## Concept

Large language models (LLMs) are text-in, text-out. Large Multi-modal Models (LMMs) generalize this beyond the text modalities. For instance, models such as GPT-4V allow you to jointly input both images and text, and output text.

We've included a base `MultiModalLLM` abstraction to allow for text+image models. **NOTE**: This naming is subject to change!

## Usage Pattern

1. The following code snippet shows how you can get started using LMMs e.g. with GPT-4V.

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

2. The following code snippet shows how you can build MultiModal Vector Stores/Index.

```python
from llama_index.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.vector_stores import QdrantVectorStore
from llama_index import SimpleDirectoryReader, StorageContext

import qdrant_client
from llama_index import (
    SimpleDirectoryReader,
)

# Create a local Qdrant vector store
client = qdrant_client.QdrantClient(path="qdrant_mm_db")

text_store = QdrantVectorStore(
    client=client, collection_name="text_collection"
)
image_store = QdrantVectorStore(
    client=client, collection_name="image_collection"
)
storage_context = StorageContext.from_defaults(vector_store=text_store)

# Create the MultiModal index
documents = SimpleDirectoryReader("./data_folder/").load_data()

index = MultiModalVectorStoreIndex.from_documents(
    documents, storage_context=storage_context, image_vector_store=image_store
)
```

3. The following code snippet shows how you can use MultiModal Retriever and Query Engine.

```python
from llama_index.multi_modal_llms import OpenAIMultiModal
from llama_index.prompts import PromptTemplate
from llama_index.query_engine import SimpleMultiModalQueryEngine

retriever_engine = index.as_retriever(
    similarity_top_k=3, image_similarity_top_k=3
)

# retrieve more information from the GPT4V response
retrieval_results = retriever_engine.retrieve(response)

qa_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_tmpl = PromptTemplate(qa_tmpl_str)

query_engine = index.as_query_engine(
    multi_modal_llm=openai_mm_llm, text_qa_template=qa_tmpl
)

query_str = "Tell me more about the Porsche"
response = query_engine.query(query_str)
```

**Legend**

- ✅ = should work fine
- ⚠️ = sometimes unreliable, may need more tuning to improve
- 🛑 = not available at the moment.

### End to End Multi-Modal Work Flow

The tables below attempt to show the **initial** steps with various LlamaIndex features for building your own Multi-Modal RAGs. You can combine different modules/steps together for composing your own Multi-Modal RAG orchestration.

| Query Type | Data Sources<br>for MultiModal<br>Vector Store/Index | MultiModal<br>Embedding                | Retriever                                        | Query<br>Engine        | Output<br>Data<br>Type                   |
| ---------- | ---------------------------------------------------- | -------------------------------------- | ------------------------------------------------ | ---------------------- | ---------------------------------------- |
| Text ✅    | Text ✅                                              | Text ✅                                | Top-k retrieval ✅<br>Simple Fusion retrieval ✅ | Simple Query Engine ✅ | Retrieved Text ✅<br>Generated Text ✅   |
| Image ✅   | Image ✅                                             | Image ✅<br>Image to Text Embedding ✅ | Top-k retrieval ✅<br>Simple Fusion retrieval ✅ | Simple Query Engine ✅ | Retrieved Image ✅<br>Generated Image 🛑 |
| Audio 🛑   | Audio 🛑                                             | Audio 🛑                               | 🛑                                               | 🛑                     | Audio 🛑                                 |
| Video 🛑   | Video 🛑                                             | Video 🛑                               | 🛑                                               | 🛑                     | Video 🛑                                 |

### Multi-Modal LLM Models

These notebooks serve as examples how to leverage and integrate Multi-Modal LLM model, Multi-Modal embeddings, Multi-Modal vector stores, Retriever, Query engine for composing Multi-Modal RAG orchestration.

| Multi-Modal<br>Vision Models                                                     | Single<br>Image<br>Reasoning | Multiple<br>Images<br>Reasoning | Image<br>Embeddings | Simple<br>Query<br>Engine | Pydantic<br>Structured<br>Output |
| -------------------------------------------------------------------------------- | ---------------------------- | ------------------------------- | ------------------- | ------------------------- | -------------------------------- |
| [GPT4V](/examples/multi_modal/gpt4v_multi_modal_retrieval.ipynb)<br>(OpenAI API) | ✅                           | ✅                              | 🛑                  | ✅                        | ✅                               |
| [CLIP](/examples/multi_modal/image_to_image_retrieval.ipynb)<br>(Local host)     | 🛑                           | 🛑                              | ✅                  | 🛑                        | 🛑                               |
| [LLaVa](/examples/multi_modal/llava_multi_modal_tesla_10q.ipynb)<br>(replicate)  | ✅                           | 🛑                              | 🛑                  | ✅                        | ⚠️                               |
| [Fuyu-8B](/examples/multi_modal/replicate_multi_modal.ipynb)<br>(replicate)      | ✅                           | 🛑                              | 🛑                  | ✅                        | ✅                               |
| [ImageBind<br>](https://imagebind.metademolab.com/)[To integrate]                | 🛑                           | 🛑                              | ✅                  | 🛑                        | 🛑                               |
| [MiniGPT-4<br>](/examples/multi_modal/replicate_multi_modal.ipynb)               | ✅                           | 🛑                              | 🛑                  | ✅                        | ⚠️                               |
| [CogVLM<br>](https://github.com/THUDM/CogVLM)                                    | ✅                           | 🛑                              | 🛑                  | ✅                        | ⚠️                               |
| [Qwen-VL<br>](https://arxiv.org/abs/2308.12966)[To integrate]                    | ✅                           | 🛑                              | 🛑                  | ✅                        | ⚠️                               |

### Multi Modal Vector Stores

Below table lists some vector stores supporting Multi-Modal use cases. Our LlamaIndex built-in `MultiModalVectorStoreIndex` supports building separate vector stores for image and text embedding vector stores. `MultiModalRetriever`, and `SimpleMultiModalQueryEngine` support text to text/image and image to image retrieval and simple ranking fusion functions for combining text and image retrieval results.
| Multi-Modal<br>Vector Stores | Single<br>Vector<br>Store | Multiple<br>Vector<br>Stores | Text<br>Embedding | Image<br>Embedding |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | --------------------------- | --------------------------------------------------------- | ------------------------------------------------------- |
| [LLamaIndex self-built<br>MultiModal Index](/examples/multi_modal/gpt4v_multi_modal_retrieval.ipynb) | 🛑 | ✅ | Can be arbitrary<br>text embedding<br>(Default is GPT3.5) | Can be arbitrary<br>Image embedding<br>(Default is CLIP) |
| [Chroma](/examples/multi_modal/ChromaMultiModalDemo.ipynb) | ✅ | 🛑 | CLIP ✅ | CLIP ✅ |
| [Weaviate](https://weaviate.io/developers/weaviate/modules/retriever-vectorizer-modules/multi2vec-bind)<br>[To integrate] | ✅ | 🛑 | CLIP ✅<br>ImageBind ✅ | CLIP ✅<br>ImageBind ✅ |

## Multi-Modal LLM Modules

We support integrations with GPT-4V, LLaVA, Fuyu-8B, CLIP, and more.

```{toctree}
---
maxdepth: 1
---
/examples/multi_modal/openai_multi_modal.ipynb
/examples/multi_modal/replicate_multi_modal.ipynb
/examples/multi_modal/multi_modal_pydantic.ipynb
/examples/multi_modal/gpt4v_experiments_cot.ipynb
/examples/multi_modal/llava_multi_modal_tesla_10q.ipynb
```

## Multi-Modal RAG

We support Multi-Modal RAG with different Multi-Modal LLMs with Multi-Modal vector stores.

```{toctree}
---
maxdepth: 1
---
/examples/multi_modal/gpt4v_multi_modal_retrieval.ipynb
/examples/multi_modal/multi_modal_pdf_tables.ipynb
/examples/multi_modal/multi_modal_retrieval.ipynb
/examples/multi_modal/image_to_image_retrieval.ipynb
/examples/multi_modal/ChromaMultiModalDemo.ipynb
```

## Evaluation

We support basic evaluation for Multi-Modal LLM and RAG.

```{toctree}
---
maxdepth: 1
---
/examples/evaluation/multi_modal/multi_modal_rag_evaluation.ipynb
```
