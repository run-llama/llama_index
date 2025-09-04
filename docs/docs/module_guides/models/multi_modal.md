# Multi-modal models

## Concept

Large language models (LLMs) are text-in, text-out. Large Multi-modal Models (LMMs) generalize this beyond the text modalities. For instance, models such as GPT-4V allow you to jointly input both images and text, and output text.

We've included a base `MultiModalLLM` abstraction to allow for text+image models. **NOTE**: This naming is subject to change!

## Usage Pattern

1. The following code snippet shows how you can get started using LMMs e.g. with GPT-4V.

```python
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.core import SimpleDirectoryReader

# load image documents from urls
image_documents = load_image_urls(image_urls)

# load image documents from local directory
image_documents = SimpleDirectoryReader(local_directory).load_data()

# non-streaming
openai_mm_llm = OpenAIMultiModal(
    model="gpt-4-vision-preview", api_key=OPENAI_API_KEY, max_new_tokens=300
)
response = openai_mm_llm.complete(
    prompt="what is in the image?", image_documents=image_documents
)
```

2. The following code snippet shows how you can build MultiModal Vector Stores/Index.

```python
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import SimpleDirectoryReader, StorageContext

import qdrant_client
from llama_index.core import SimpleDirectoryReader

# Create a local Qdrant vector store
client = qdrant_client.QdrantClient(path="qdrant_mm_db")

# if you only need image_store for image retrieval,
# you can remove text_store
text_store = QdrantVectorStore(
    client=client, collection_name="text_collection"
)
image_store = QdrantVectorStore(
    client=client, collection_name="image_collection"
)

storage_context = StorageContext.from_defaults(
    vector_store=text_store, image_store=image_store
)

# Load text and image documents from local folder
documents = SimpleDirectoryReader("./data_folder/").load_data()
# Create the MultiModal index
index = MultiModalVectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)
```

3. The following code snippet shows how you can use MultiModal Retriever and Query Engine.

```python
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import SimpleMultiModalQueryEngine

retriever_engine = index.as_retriever(
    similarity_top_k=3, image_similarity_top_k=3
)

# retrieve more information from the GPT4V response
retrieval_results = retriever_engine.retrieve(response)

# if you only need image retrieval without text retrieval
# you can use `text_to_image_retrieve`
# retrieval_results = retriever_engine.text_to_image_retrieve(response)

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

The tables below attempt to show the **initial** steps with various LlamaIndex features for building your own Multi-Modal RAGs (Retrieval Augmented Generation). You can combine different modules/steps together for composing your own Multi-Modal RAG orchestration.

| Query Type | Data Sources<br>for MultiModal<br>Vector Store/Index | MultiModal<br>Embedding                | Retriever                                        | Query<br>Engine        | Output<br>Data<br>Type                   |
| ---------- | ---------------------------------------------------- | -------------------------------------- | ------------------------------------------------ | ---------------------- | ---------------------------------------- |
| Text ✅    | Text ✅                                              | Text ✅                                | Top-k retrieval ✅<br>Simple Fusion retrieval ✅ | Simple Query Engine ✅ | Retrieved Text ✅<br>Generated Text ✅   |
| Image ✅   | Image ✅                                             | Image ✅<br>Image to Text Embedding ✅ | Top-k retrieval ✅<br>Simple Fusion retrieval ✅ | Simple Query Engine ✅ | Retrieved Image ✅<br>Generated Image 🛑 |
| Audio 🛑   | Audio 🛑                                             | Audio 🛑                               | 🛑                                               | 🛑                     | Audio 🛑                                 |
| Video 🛑   | Video 🛑                                             | Video 🛑                               | 🛑                                               | 🛑                     | Video 🛑                                 |

### Multi-Modal LLM Models

These notebooks serve as examples how to leverage and integrate Multi-Modal LLM model, Multi-Modal embeddings, Multi-Modal vector stores, Retriever, Query engine for composing Multi-Modal Retrieval Augmented Generation (RAG) orchestration.

| Multi-Modal<br>Vision Models                                                            | Single<br>Image<br>Reasoning | Multiple<br>Images<br>Reasoning | Image<br>Embeddings | Simple<br>Query<br>Engine | Pydantic<br>Structured<br>Output |
| --------------------------------------------------------------------------------------- | ---------------------------- | ------------------------------- | ------------------- | ------------------------- | -------------------------------- |
| [GPT4V](../../examples/multi_modal/gpt4v_multi_modal_retrieval.ipynb)<br>(OpenAI API)   | ✅                           | ✅                              | 🛑                  | ✅                        | ✅                               |
| [GPT4V-Azure](../../examples/multi_modal/azure_openai_multi_modal.ipynb)<br>(Azure API) | ✅                           | ✅                              | 🛑                  | ✅                        | ✅                               |
| [Gemini](../../examples/multi_modal/gemini.ipynb)<br>(Google)                           | ✅                           | ✅                              | 🛑                  | ✅                        | ✅                               |
| [CLIP](../../examples/multi_modal/image_to_image_retrieval.ipynb)<br>(Local host)       | 🛑                           | 🛑                              | ✅                  | 🛑                        | 🛑                               |
| [LLaVa](../../examples/multi_modal/llava_multi_modal_tesla_10q.ipynb)<br>(replicate)    | ✅                           | 🛑                              | 🛑                  | ✅                        | ⚠️                               |
| [Fuyu-8B](../../examples/multi_modal/replicate_multi_modal.ipynb)<br>(replicate)        | ✅                           | 🛑                              | 🛑                  | ✅                        | ⚠️                               |
| [ImageBind<br>](https://imagebind.metademolab.com/)[To integrate]                       | 🛑                           | 🛑                              | ✅                  | 🛑                        | 🛑                               |
| [MiniGPT-4<br>](../../examples/multi_modal/replicate_multi_modal.ipynb)                 | ✅                           | 🛑                              | 🛑                  | ✅                        | ⚠️                               |
| [CogVLM<br>](https://github.com/THUDM/CogVLM)                                           | ✅                           | 🛑                              | 🛑                  | ✅                        | ⚠️                               |
| [Qwen-VL<br>](https://arxiv.org/abs/2308.12966)[To integrate]                           | ✅                           | 🛑                              | 🛑                  | ✅                        | ⚠️                               |

### Multi Modal Vector Stores

Below table lists some vector stores supporting Multi-Modal use cases. Our LlamaIndex built-in `MultiModalVectorStoreIndex` supports building separate vector stores for image and text embedding vector stores. `MultiModalRetriever`, and `SimpleMultiModalQueryEngine` support text to text/image and image to image retrieval and simple ranking fusion functions for combining text and image retrieval results.
| Multi-Modal<br>Vector Stores | Single<br>Vector<br>Store | Multiple<br>Vector<br>Stores | Text<br>Embedding | Image<br>Embedding |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | --------------------------- | --------------------------------------------------------- | ------------------------------------------------------- |
| [LLamaIndex self-built<br>MultiModal Index](../../examples/multi_modal/gpt4v_multi_modal_retrieval.ipynb) | 🛑 | ✅ | Can be arbitrary<br>text embedding<br>(Default is GPT3.5) | Can be arbitrary<br>Image embedding<br>(Default is CLIP) |
| [Chroma](../../examples/multi_modal/ChromaMultiModalDemo.ipynb) | ✅ | 🛑 | CLIP ✅ | CLIP ✅ |
| [Weaviate](https://weaviate.io/developers/weaviate/modules/retriever-vectorizer-modules/multi2vec-bind)<br>[To integrate] | ✅ | 🛑 | CLIP ✅<br>ImageBind ✅ | CLIP ✅<br>ImageBind ✅ |

## Multi-Modal LLM Modules

We support integrations with GPT4-V, Anthropic (Opus, Sonnet), Gemini (Google), CLIP (OpenAI), BLIP (Salesforce), and Replicate (LLaVA, Fuyu-8B, MiniGPT-4, CogVLM), and more.

- [OpenAI](../../examples/multi_modal/openai_multi_modal.ipynb)
- [Gemini](../../examples/multi_modal/gemini.ipynb)
- [Anthropic](../../examples/multi_modal/anthropic_multi_modal.ipynb)
- [Replicate](../../examples/multi_modal/replicate_multi_modal.ipynb)
- [Pydantic Multi-Modal](../../examples/multi_modal/multi_modal_pydantic.ipynb)
- [GPT-4v COT Experiments](../../examples/multi_modal/gpt4v_experiments_cot.ipynb)
- [Llava Tesla 10q](../../examples/multi_modal/llava_multi_modal_tesla_10q.ipynb)

## Multi-Modal Retrieval Augmented Generation

We support Multi-Modal Retrieval Augmented Generation with different Multi-Modal LLMs with Multi-Modal vector stores.

- [GPT-4v Retrieval](../../examples/multi_modal/gpt4v_multi_modal_retrieval.ipynb)
- [Multi-Modal Retrieval](../../examples/multi_modal/multi_modal_retrieval.ipynb)
- [Image-to-Image Retrieval](../../examples/multi_modal/image_to_image_retrieval.ipynb)
- [Chroma Multi-Modal](../../examples/multi_modal/ChromaMultiModalDemo.ipynb)

## Evaluation

We support basic evaluation for Multi-Modal LLM and Retrieval Augmented Generation.

- [Multi-Modal RAG Eval](../../examples/evaluation/multi_modal/multi_modal_rag_evaluation.ipynb)
