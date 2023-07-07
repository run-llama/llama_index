# Evaluating and Tracking with TruLens

This page covers how to use [TruLens](https://trulens.org) to evaluate and track LLM apps built on Llama-Index.

## What is TruLens?

TruLens is an [opensource](https://github.com/truera/trulens) package that provides instrumentation and evaluation tools for large language model (LLM) based applications. This includes feedback function evaluations of relevance, sentiment and more, plus in-depth tracing including cost and latency.

![TruLens Architecture](https://github.com/truera/trulens/blob/main/docs/Assets/image/TruLens_Architecture.png)

As you iterate on new versions of your LLM application, you can compare their performance across all of the different quality metrics you've set up. You'll also be able to view evaluations at a record level, and explore the app metadata for each record.

### Installation and Setup

Adding TruLens is simple, just install it from pypi!

```sh
pip install trulens-eval
```

```python
from trulens_eval import TruLlama

```

## Try it out!

[llama_index_quickstart.ipynb](https://github.com/truera/trulens/blob/releases/rc-trulens-eval-0.4.0/trulens_eval/examples/frameworks/llama_index/llama_index_quickstart.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/google-colab/trulens_eval/examples/colab/quickstarts/llama_index_quickstart_colab.ipynb)

## Read more

* [Build and Evaluate LLM Apps with LlamaIndex and TruLens](https://medium.com/llamaindex-blog/build-and-evaluate-llm-apps-with-llamaindex-and-trulens-6749e030d83c)

* [trulens.org](https://www.trulens.org/)
