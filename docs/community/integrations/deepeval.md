# Unit Testing LLMs With DeepEval

[DeepEval](https://github.com/confident-ai/deepeval) provides unit testing for AI agents and LLM-powered applications. It provides a really simple interface for LlamaIndex developers to write tests and helps developers ensure AI applications run as expected.

DeepEval provides an opinionated framework to measure responses and is completely open-source.

### Installation and Setup

Adding [DeepEval](https://github.com/confident-ai/deepeval) is simple, just install and configure it:

```sh
pip install -q -q llama-index
pip install -U deepeval
```

Once installed , you can get set up and start writing tests.

```sh
# Optional step: Login to get a nice dashboard for your tests later!
# During this step - make sure to save your project as llama
deepeval login
deepeval test generate test_sample.py
```

You can then run tests as such:

```bash
deepeval test run test_sample.py
```

After running this, you will get a beautiful dashboard like so:

![Sample dashboard](https://raw.githubusercontent.com/confident-ai/deepeval/main/docs/assets/dashboard-screenshot.png)

## Types of Tests

DeepEval presents an opinionated framework for the types of tests that are being run. It breaks down LLM outputs into: 
- Answer Relevancy - [Read more here](https://docs.confident-ai.com/docs/measuring_llm_performance/answer_relevancy)
- Factual Consistency (to measure the extent of hallucinations) - [Read more here](https://docs.confident-ai.com/docs/measuring_llm_performance/factual_consistency)
- Conceptual Similarity (to know if answers are in line with expectations) - [Read more here](https://docs.confident-ai.com/docs/measuring_llm_performance/conceptual_similarity)
- Toxicness - [Read more here](https://docs.confident-ai.com/docs/measuring_llm_performance/non_toxic)
- Bias (can come up from finetuning) - [Read more here](https://docs.confident-ai.com/docs/measuring_llm_performance/debias)

You can more about the [DeepEval Framework](https://docs.confident-ai.com/docs/framework) here.

## Use With Your LlamaIndex

DeepEval integrates nicely with LlamaIndex's `ResponseEvaluator` class. Below is an example of the factual consistency documentation.

```python

from llama_index.response.schema import Response
from typing import List
from llama_index.schema import Document
from deepeval.metrics.factual_consistency import FactualConsistencyMetric

from llama_index import (
    TreeIndex,
    VectorStoreIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    Response,
)
from llama_index.llms import OpenAI
from llama_index.evaluation import ResponseEvaluator

import os
import openai

api_key = "sk-XXX"
openai.api_key = api_key

gpt4 = OpenAI(temperature=0, model="gpt-4", api_key=api_key)
service_context_gpt4 = ServiceContext.from_defaults(llm=gpt4)
evaluator_gpt4 = ResponseEvaluator(service_context=service_context_gpt4)

```

#### Getting a lLamaHub Loader 

```python
from llama_index import download_loader

WikipediaReader = download_loader("WikipediaReader")

loader = WikipediaReader()
documents = loader.load_data(pages=['Tokyo'])
tree_index = TreeIndex.from_documents(documents=documents)
vector_index = VectorStoreIndex.from_documents(
    documents, service_context=service_context_gpt4
)
```

We then build an evaluator based on the `BaseEvaluator` class that requires an `evaluate` method.

In this example, we show you how to write a factual consistency check.

```python
class FactualConsistencyResponseEvaluator:
  def get_context(self, response: Response) -> List[Document]:
    """Get context information from given Response object using source nodes.

    Args:
        response (Response): Response object from an index based on the query.

    Returns:
        List of Documents of source nodes information as context information.
    """
    context = []

    for context_info in response.source_nodes:
        context.append(Document(text=context_info.node.get_content()))

    return context

  def evaluate(self, response: Response) -> str:
    """Evaluate factual consistency metrics
    """
    answer = str(response)
    context = self.get_context(response)
    metric = FactualConsistencyMetric()
    context = " ".join([d.text for d in context])
    score = metric.measure(output=answer, context=context)
    if metric.is_successful():
        return "YES"
    else:
        return "NO"

evaluator = FactualConsistencyResponseEvaluator()
```

You can then evaluate as such:

```python
query_engine = tree_index.as_query_engine()
response = query_engine.query("How did Tokyo get its name?")
eval_result = evaluator.evaluate(response)
```

### Useful Links

* [Read About The DeepEval Framework](https://docs.confident-ai.com/docs/framework)
* [Answer Relevancy](https://docs.confident-ai.com/docs/measuring_llm_performance/answer_relevancy)
* [Conceptual Similarity](https://docs.confident-ai.com/docs/measuring_llm_performance/conceptual_similarity) . 
* [Bias](https://docs.confident-ai.com/docs/measuring_llm_performance/debias)