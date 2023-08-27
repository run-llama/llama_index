# Unit Testing LLMs With DeepEval

[DeepEval](https://github.com/confident-ai/deepeval) provides unit testing for AI agents and LLM-powered applications. It provides a really simple interface for LlamaIndex developers to write tests and helps developers ensure AI applications run as expected.

DeepEval provides an opinionated framework to measure responses and is completely open-source.

### Installation and Setup

Adding [DeepEval](https://github.com/confident-ai/deepeval) is simple, just install and configure it:

```sh
pip install deepeval
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

## LlamaIndex Integration

DeepEval integrates nicely with LlamaIndex's `ResponseEvaluator` class. Below is an example of the factual consistency documentation.

```python
from llama_index.response.schema import Response
from deepeval.metrics.factual_consistency import FactualConsistencyMetric

class DeepEvalResponseEvaluator:
    """Response Evaluator based on DeepEval framework.
    """
    def evaluate(self, response: Response) -> str:
        """
        """
        answer = str(response)
        context = self.get_context(response)
        metric = FactualConsistencyMetric()
        score = metric.measure(output=answer, context=context)
        if metric.is_successful():
            return "YES"
        else:
            return "NO"

```

### Useful Links

* [Read About The DeepEval Framework](https://docs.confident-ai.com/docs/framework)
* [Answer Relevancy](https://docs.confident-ai.com/docs/measuring_llm_performance/answer_relevancy)
* [Conceptual Similarity](https://docs.confident-ai.com/docs/measuring_llm_performance/conceptual_similarity)