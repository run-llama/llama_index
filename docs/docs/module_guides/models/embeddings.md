# Embeddings

## Concept

Embeddings are used in LlamaIndex to represent your documents using a sophisticated numerical representation. Embedding models take text as input, and return a long list of numbers used to capture the semantics of the text. These embedding models have been trained to represent text this way, and help enable many applications, including search!

At a high level, if a user asks a question about dogs, then the embedding for that question will be highly similar to text that talks about dogs.

When calculating the similarity between embeddings, there are many methods to use (dot product, cosine similarity, etc.). By default, LlamaIndex uses cosine similarity when comparing embeddings.

There are many embedding models to pick from. By default, LlamaIndex uses `text-embedding-ada-002` from OpenAI. We also support any embedding model offered by Langchain [here](https://python.langchain.com/docs/modules/data_connection/text_embedding/), as well as providing an easy to extend base class for implementing your own embeddings.

## Usage Pattern

Most commonly in LlamaIndex, embedding models will be specified in the `Settings` object, and then used in a vector index. The embedding model will be used to embed the documents used during index construction, as well as embedding any queries you make using the query engine later on. You can also specify embedding models per-index.

If you don't already have your embeddings installed:

```bash
pip install llama-index-embeddings-openai
```

Then:

```python
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings

# changing the global default
Settings.embed_model = OpenAIEmbedding()

# local usage
embedding = OpenAIEmbedding().get_text_embedding("hello world")
embeddings = OpenAIEmbedding().get_text_embeddings(
    ["hello world", "hello world"]
)

# per-index
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
```

To save costs, you may want to use a local model.

```bash
pip install llama-index-embeddings-huggingface
```

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
```

This will use a well-performing and fast default from [Hugging Face](https://huggingface.co/models?library=sentence-transformers).

You can find more usage details and available customization options below.

## Getting Started

The most common usage for an embedding model will be setting it in the global `Settings` object, and then using it to construct an index and query. The input documents will be broken into nodes, and the embedding model will generate an embedding for each node.

By default, LlamaIndex will use `text-embedding-ada-002`, which is what the example below manually sets up for you.

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

# global default
Settings.embed_model = OpenAIEmbedding()

documents = SimpleDirectoryReader("./data").load_data()

index = VectorStoreIndex.from_documents(documents)
```

Then, at query time, the embedding model will be used again to embed the query text.

```python
query_engine = index.as_query_engine()

response = query_engine.query("query string")
```

## Customization

### Batch Size

By default, embeddings requests are sent to OpenAI in batches of 10. For some users, this may (rarely) incur a rate limit. For other users embedding many documents, this batch size may be too small.

```python
# set the batch size to 42
embed_model = OpenAIEmbedding(embed_batch_size=42)
```

### Local Embedding Models

The easiest way to use a local model is by using [`HuggingFaceEmbedding`](https://docs.llamaindex.ai/en/stable/api_reference/embeddings/huggingface/#llama_index.embeddings.huggingface.HuggingFaceEmbedding) from `llama-index-embeddings-huggingface`:

```python
# pip install llama-index-embeddings-huggingface
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
```

Which loads the [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) embedding model. You can use [any Sentence Transformers embedding model from Hugging Face](https://huggingface.co/models?library=sentence-transformers).

Beyond the keyword arguments available in the [`HuggingFaceEmbedding`](https://docs.llamaindex.ai/en/stable/api_reference/embeddings/huggingface/#llama_index.embeddings.huggingface.HuggingFaceEmbedding) constructor, additional keyword arguments are passed down to the underlying [`SentenceTransformer` instance](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html), like `backend`, `model_kwargs`, `truncate_dim`, `revision`, etc.

### ONNX or OpenVINO optimizations

LlamaIndex also supports using ONNX or OpenVINO to speed up local inference, by relying on [Sentence Transformers](https://sbert.net) and [Optimum](https://huggingface.co/docs/optimum/index).

Some prerequisites:

```bash
pip install llama-index-embeddings-huggingface
# Plus any of the following:
pip install optimum[onnxruntime-gpu] # For ONNX on GPUs
pip install optimum[onnxruntime]     # For ONNX on CPUs
pip install optimum-intel[openvino]  # For OpenVINO
```

Creation with specifying the model and output path:

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    backend="onnx",  # or "openvino"
)
```

If the model repository does not already contain an ONNX or OpenVINO model, then it will be automatically converted using Optimum.
See the [Sentence Transformers documentation](https://sbert.net/docs/sentence_transformer/usage/efficiency.html#benchmarks) for benchmarks of the various options.

<details><summary>What if I want to use an optimized or quantized model checkpoint instead?</summary>
It's common for embedding models to have multiple ONNX and/or OpenVINO checkpoints, for example <a href="https://huggingface.co/sentence-transformers/all-mpnet-base-v2/tree/main">sentence-transformers/all-mpnet-base-v2</a> with <a href="https://huggingface.co/sentence-transformers/all-mpnet-base-v2/tree/main/openvino">2 OpenVINO checkpoints</a> and <a href="https://huggingface.co/sentence-transformers/all-mpnet-base-v2/tree/main/onnx">9 ONNX checkpoints</a>. See the <a href="https://sbert.net/docs/sentence_transformer/usage/efficiency.html">Sentence Transformers documentation</a> for more details on each of these options and their expected performance.

You can specify a <code>file_name</code> in the <code>model_kwargs</code> argument to load a specific checkpoint. For example, to load the <code>openvino/openvino_model_qint8_quantized.xml</code> checkpoint from the <code>sentence-transformers/all-mpnet-base-v2</code> model repository:

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

quantized_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2",
    backend="openvino",
    device="cpu",
    model_kwargs={"file_name": "openvino/openvino_model_qint8_quantized.xml"},
)
Settings.embed_model = quantized_model
```

</details>

<details><summary>What option should I use on CPUs?</summary>

As shown in the Sentence Transformers benchmarks, OpenVINO quantized to int8 (<code>openvino_model_qint8_quantized.xml</code>) is extremely performant, at a small cost to accuracy. If you want to ensure identical results, then the basic <code>backend="openvino"</code> or <code>backend="onnx"</code> might be the strongest options.

<img src="https://sbert.net/_images/backends_benchmark_cpu.png" alt="CPU Backend Benchmark">

Given this query and these documents, the following results were obtained with int8 quantized OpenVINO versus the default Hugging Face model:
```python
query = "Which planet is known as the Red Planet?"
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
]
```

```
HuggingFaceEmbedding(device='cpu'):
- Average throughput: 38.20 queries/sec (over 5 runs)
- Query-document similarities tensor([[0.7783, 0.4654, 0.6919, 0.7010]])

HuggingFaceEmbedding(backend='openvino', device='cpu', model_kwargs={'file_name': 'openvino_model_qint8_quantized.xml'}):
- Average throughput: 266.08 queries/sec (over 5 runs)
- Query-document similarities tensor([[0.7492, 0.4623, 0.6606, 0.6556]])
```

That's a 6.97x speedup while keeping the same document ranking.

<details><summary>Click to see a reproduction script</summary>

```python
import time
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

quantized_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2",
    backend="openvino",
    device="cpu",
    model_kwargs={"file_name": "openvino/openvino_model_qint8_quantized.xml"},
)
quantized_model_desc = "HuggingFaceEmbedding(backend='openvino', device='cpu', model_kwargs={'file_name': 'openvino_model_qint8_quantized.xml'})"
baseline_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2",
    device="cpu",
)
baseline_model_desc = "HuggingFaceEmbedding(device='cpu')"

query = "Which planet is known as the Red Planet?"


def bench(model, query, description):
    for _ in range(3):
        model.get_agg_embedding_from_queries([query] * 32)

    sentences_per_second = []
    for _ in range(5):
        queries = [query] * 512
        start_time = time.time()
        model.get_agg_embedding_from_queries(queries)
        sentences_per_second.append(len(queries) / (time.time() - start_time))

    print(
        f"{description:<120}: Avg throughput: {sum(sentences_per_second) / len(sentences_per_second):.2f} queries/sec (over 5 runs)"
    )


bench(baseline_model, query, baseline_model_desc)
bench(quantized_model, query, quantized_model_desc)

# Example documents for similarity comparison. The first is the correct one, and the rest are distractors.
docs = [
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
]

baseline_query_embedding = baseline_model.get_query_embedding(query)
baseline_doc_embeddings = baseline_model.get_text_embedding_batch(docs)

quantized_query_embedding = quantized_model.get_query_embedding(query)
quantized_doc_embeddings = quantized_model.get_text_embedding_batch(docs)

baseline_similarity = baseline_model._model.similarity(
    baseline_query_embedding, baseline_doc_embeddings
)
print(
    f"{baseline_model_desc:<120}: Query-document similarities {baseline_similarity}"
)
quantized_similarity = quantized_model._model.similarity(
    quantized_query_embedding, quantized_doc_embeddings
)
print(
    f"{quantized_model_desc:<120}: Query-document similarities {quantized_similarity}"
)
```

</details>

</details>

<details><summary>What option should I use on GPUs?</summary>

On GPUs, OpenVINO is not particularly interesting, and ONNX does not necessarily outperform a quantized model running in the default <code>torch</code> backend.

<img src="https://sbert.net/_images/backends_benchmark_gpu.png" alt="GPU Backend Benchmark">

That means that you don't need additional dependencies for a strong speedup on GPUs, you can just use a lower precision when loading the model:

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device="cuda",
    model_kwargs={"torch_dtype": "float16"},
)
```

</details>

<details><summary>What if my desired model does not have my desired backend and optimization or quantization?</summary>

The <a href="https://huggingface.co/spaces/sentence-transformers/backend-export">backend-export</a> Hugging Face Space can be used to convert any Sentence Transformers model to ONNX or OpenVINO with quantization or optimization applied. This will create a pull request to the model repository with the converted model files. You can then use this model in LlamaIndex by specifying the <code>revision</code> argument like so:

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    backend="openvino",
    revision="refs/pr/16",  # for pull request 16: https://huggingface.co/BAAI/bge-small-en-v1.5/discussions/16
    model_kwargs={"file_name": "openvino_model_qint8_quantized.xml"},
)
```

</details>

### LangChain Integrations

We also support any embeddings offered by Langchain [here](https://python.langchain.com/docs/modules/data_connection/text_embedding/).

The example below loads a model from Hugging Face, using Langchain's embedding class.

```
pip install llama-index-embeddings-langchain
```

```python
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from llama_index.core import Settings

Settings.embed_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en")
```

### Custom Embedding Model

If you wanted to use embeddings not offered by LlamaIndex or Langchain, you can also extend our base embeddings class and implement your own!

The example below uses Instructor Embeddings ([install/setup details here](https://huggingface.co/hkunlp/instructor-large)), and implements a custom embeddings class. Instructor embeddings work by providing text, as well as "instructions" on the domain of the text to embed. This is helpful when embedding text from a very specific and specialized topic.

```python
from typing import Any, List
from InstructorEmbedding import INSTRUCTOR
from llama_index.core.embeddings import BaseEmbedding


class InstructorEmbeddings(BaseEmbedding):
    def __init__(
        self,
        instructor_model_name: str = "hkunlp/instructor-large",
        instruction: str = "Represent the Computer Science documentation or question:",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model = INSTRUCTOR(instructor_model_name)
        self._instruction = instruction

        def _get_query_embedding(self, query: str) -> List[float]:
            embeddings = self._model.encode([[self._instruction, query]])
            return embeddings[0]

        def _get_text_embedding(self, text: str) -> List[float]:
            embeddings = self._model.encode([[self._instruction, text]])
            return embeddings[0]

        def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
            embeddings = self._model.encode(
                [[self._instruction, text] for text in texts]
            )
            return embeddings

        async def _get_query_embedding(self, query: str) -> List[float]:
            return self._get_query_embedding(query)

        async def _get_text_embedding(self, text: str) -> List[float]:
            return self._get_text_embedding(text)
```

## Standalone Usage

You can also use embeddings as a standalone module for your project, existing application, or general testing and exploration.

```python
embeddings = embed_model.get_text_embedding(
    "It is raining cats and dogs here!"
)
```

## List of supported embeddings

We support integrations with OpenAI, Azure, and anything LangChain offers.

- [Azure OpenAI](../../examples/customization/llms/AzureOpenAI.ipynb)
- [CalrifAI](../../examples/embeddings/clarifai.ipynb)
- [Cohere](../../examples/embeddings/cohereai.ipynb)
- [Custom](../../examples/embeddings/custom_embeddings.ipynb)
- [Dashscope](../../examples/embeddings/dashscope_embeddings.ipynb)
- [ElasticSearch](../../examples/embeddings/elasticsearch.ipynb)
- [FastEmbed](../../examples/embeddings/fastembed.ipynb)
- [Google Palm](../../examples/embeddings/google_palm.ipynb)
- [Gradient](../../examples/embeddings/gradient.ipynb)
- [Anyscale](../../examples/embeddings/Anyscale.ipynb)
- [Huggingface](../../examples/embeddings/huggingface.ipynb)
- [JinaAI](../../examples/embeddings/jinaai_embeddings.ipynb)
- [Langchain](../../examples/embeddings/Langchain.ipynb)
- [LLM Rails](../../examples/embeddings/llm_rails.ipynb)
- [MistralAI](../../examples/embeddings/mistralai.ipynb)
- [OpenAI](../../examples/embeddings/OpenAI.ipynb)
- [Sagemaker](../../examples/embeddings/sagemaker_embedding_endpoint.ipynb)
- [Text Embedding Inference](../../examples/embeddings/text_embedding_inference.ipynb)
- [TogetherAI](../../examples/embeddings/together.ipynb)
- [Upstage](../../examples/embeddings/upstage.ipynb)
- [VoyageAI](../../examples/embeddings/voyageai.ipynb)
- [Nomic](../../examples/embeddings/nomic.ipynb)
- [Fireworks AI](../../examples/embeddings/fireworks.ipynb)
