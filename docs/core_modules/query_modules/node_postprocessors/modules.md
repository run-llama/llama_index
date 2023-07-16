# Modules

## SimilarityPostprocessor

Used to remove nodes that are below a similarity score threshold.

```python
from llama_index.indices.postprocessor import SimilarityPostprocessor

postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)

postprocessor.postprocess_nodes(nodes)
```

## KeywordNodePostprocessor

Used to ensure certain keywords are either excluded or included.

```python
from llama_index.indices.postprocessor import KeywordNodePostprocessor

postprocessor = KeywordNodePostprocessor(
  required_keywords=["word1", "word2"],
  exclude_keywords=["word3", "word4"]
)

postprocessor.postprocess_nodes(nodes)
```

## SentenceEmbeddingOptimizer

This postprocessor optimizes token usage by removing sentences that are not relevant to the query (this is done using embeddings).

The percentile cutoff is a measure for using the top percentage of relevant sentences.

The threshold cutoff can be specified instead, which uses a raw similarity cutoff for picking which sentences to keep.

```python
from llama_index.indices.postprocessor import SentenceEmbeddingOptimizer

postprocessor = SentenceEmbeddingOptimizer(
  embed_model=service_context.embed_model,
  percentile_cutoff=0.5,
  # threshold_cutoff=0.7
)

postprocessor.postprocess_nodes(nodes)
```

A full notebook guide can be found [here](/examples/node_postprocessor/OptimizerDemo.ipynb)

## CohereRerank

Uses the "Cohere ReRank" functionality to re-order nodes, and returns the top N nodes.

```python
from llama_index.indices import CohereRerank

postprocessor = CohereRerank(
  top_n=2
  model="rerank-english-v2.0",
  api_key="YOUR COHERE API KEY"
)

postprocessor.postprocess_nodes(nodes)
```

## SentenceTransformerRerank

Uses the cross-encoders from the `sentence-transformer` package to re-order nodes, and returns the top N nodes.

```python
from llama_index.indices.postprocessor import CohereRerank

postprocessor = SentenceTransformerRerank(
  model="cross-encoder/ms-marco-MiniLM-L-2-v2", 
  top_n=3
)

postprocessor.postprocess_nodes(nodes)
```

Full notebook guide is available [here](/docs/examples/node_postprocessor/SentenceTransformerRerank.ipynb).

## LLM Rerank

Uses a LLM to re-order nodes by asking the LLM to return the relevant documents and a score of how relevant they are. Returns the top N ranked nodes.

```python
from llama_index.indices.postprocessor import LLMRerank

postprocessor = LLMRerank(
  top_n=2
  service_context=service_context,
)

postprocessor.postprocess_nodes(nodes)
```

Full notebook guide is available [her for Gatsby](/examples/node_postprocessor/LLMReranker-Gatsby.ipynb) and [here for Lyft 10K documents](/examples/node_postprocessor/LLMReranker-Lyft-10k.ipynb).

## FixedRecencyPostprocessor

This postproccesor returns the top K nodes sorted by date. This assumes there is a `date` field to parse in the metadata of each node.

```python
from llama_index.indices.postprocessor import FixedRecencyPostprocessor

postprocessor = FixedRecencyPostprocessor(
  tok_k=1,
  date_key="date"  # the key in the metadata to find the date
)

postprocessor.postprocess_nodes(nodes)
```

![](/_static/node_postprocessors/recency.png)

A full notebook guide is available [here](/examples/node_postprocessor/RecencyPostprocessorDemo.ipynb).

## EmbeddingRecencyPostprocessor

This postproccesor returns the top K nodes after sorting by date and removing older nodes that are too similar after measuring embedding similarity.

```python
from llama_index.indices.postprocessor import EmbeddingRecencyPostprocessor

postprocessor = EmbeddingRecencyPostprocessor(
  service_context=service_context,
  date_key="date",
  similarity_cutoff=0.7
)

postprocessor.postprocess_nodes(nodes)
```

A full notebook guide is available [here](/examples/node_postprocessor/RecencyPostprocessorDemo.ipynb).

## TimeWeightedPostprocessor

This postproccesor returns the top K nodes applying a time-weighted rerank to each node. Each time a node is retrieved, the time it was retrieved is recorded. This biases search to favor information that has not be returned in a query yet.

```python
from llama_index.indices.postprocessor import TimeWeightedPostprocessor

postprocessor = TimeWeightedPostprocessor(
  time_decay=0.99,
  top_k=1
)

postprocessor.postprocess_nodes(nodes)
```

A full notebook guide is available [here](/examples/node_postprocessor/TimeWeightedPostprocessorDemo.ipynb).

## (Beta) PIINodePostprocessor

The PII (Personal Identifiable Information) postprocssor removes information that might be a security risk. It does this by using NER (either with a dedicated NER model, or with a local LLM model).

### LLM Version

```python
from llama_index.indices.postprocessor import PIINodePostprocessor

postprocessor = PIINodePostprocessor(
  service_context=service_context,  # this should be setup with an LLM you trust
)

postprocessor.postprocess_nodes(nodes)
```

### NER Version

This version uses the default local model from Hugging Face that is loaded when you run `pipline("ner")`.

```python
from llama_index.indices.postprocessor import NERPIINodePostprocessor

postprocessor = NERPIINodePostprocessor()

postprocessor.postprocess_nodes(nodes)
```

A full notebook guide for both can be found [here](/examples/node_postprocessor/PII.ipynb).

## (Beta) PrevNextNodePostprocessor

Uses pre-defined settings to read the `Node` relationships and fetch either all nodes that come previously, next, or both.

This is useful when you know the relationships point to important data (either before, after, or both) that should be sent to the LLM if that node is retrieved.

```python
from llama_index.indices.postprocessor import PrevNextNodePostprocessor

postprocessor = PrevNextNodePostprocessor(
  docstore=index.docstore,
  num_nodes=1,  # number of nodes to fetch when looking forawrds or backwards
  mode="next"   # can be either 'next', 'previous', or 'both'
)

postprocessor.postprocess_nodes(nodes)
```

![](/_static/node_postprocessors/prev_next.png)

## (Beta) AutoPrevNextNodePostprocessor

The same as PrevNextNodePostprocessor, but lets the LLM decide the mode (next, previous, or both).

```python
from llama_index.indices.postprocessor import AutoPrevNextNodePostprocessor

postprocessor = AutoPrevNextNodePostprocessor(
  docstore=index.docstore,
  service_context=service_context
  num_nodes=1,  # number of nodes to fetch when looking forawrds or backwards)

postprocessor.postprocess_nodes(nodes)
```

A full example notebook is available [here](/examples/node_postprocessor/PrevNextPostprocessorDemo.ipynb).

## All Notebooks

```{toctree}
---
maxdepth: 1
---
/examples/node_postprocessor/OptimizerDemo.ipynb
/examples/node_postprocessor/CohereRerank.ipynb
/examples/node_postprocessor/LLMReranker-Lyft-10k.ipynb
/examples/node_postprocessor/LLMReranker-Gatsby.ipynb
/examples/node_postprocessor/RecencyPostprocessorDemo.ipynb
/examples/node_postprocessor/TimeWeightedPostprocessorDemo.ipynb
/examples/node_postprocessor/PII.ipynb
/examples/node_postprocessor/PrevNextPostprocessorDemo.ipynb
```