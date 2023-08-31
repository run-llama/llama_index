(evaluation)=
# Evaluation

## Concept
Evaluation in generative AI and retrieval is a difficult task. Due to the unpredictable nature of text, and a general lack of "expected" outcomes to compare against, there are many blockers to getting started with evaluation.

However, LlamaIndex offers a few key modules for evaluating the quality of both Document retrieval and response synthesis.
Here are some key questions for each component:

- **Document retrieval**: Are the sources relevant to the query?
- **Response synthesis**: Does the response match the retrieved context? Does it also match the query? 

This guide describes how the evaluation components within LlamaIndex work. Note that our current evaluation modules
do *not* require ground-truth labels. Evaluation can be done with some combination of the query, context, response,
and combine these with LLM calls.

### Evaluation of the Response + Context

Each response from a `query_engine.query` calls returns both the synthesized response as well as source documents.

We can evaluate the response against the retrieved sources - without taking into account the query!

This allows you to measure hallucination - if the response does not match the retrieved sources, this means that the model may be "hallucinating" an answer since it is not rooting the answer in the context provided to it in the prompt.

There are two sub-modes of evaluation here. We can either get a binary response "YES"/"NO" on whether response matches *any* source context,
and also get a response list across sources to see which sources match.

The `ResponseEvaluator` handles both modes for evaluating in this context.

### Evaluation of the Query + Response + Source Context

This is similar to the above section, except now we also take into account the query. The goal is to determine if
the response + source context answers the query.

As with the above, there are two submodes of evaluation. 
- We can either get a binary response "YES"/"NO" on whether
the response matches the query, and whether any source node also matches the query.
- We can also ignore the synthesized response, and check every source node to see
if it matches the query.

### Question Generation

In addition to evaluating queries, LlamaIndex can also use your data to generate questions to evaluate on. This means that you can automatically generate questions, and then run an evaluation pipeline to test if the LLM can actually answer questions accurately using your data.

## Integrations

We also integrate with community evaluation tools.

- [DeepEval](../../../community/integrations/deepeval.md)
- [Ragas](https://github.com/explodinggradients/ragas/blob/main/docs/integrations/llamaindex.ipynb)

## Usage Pattern

For full usage details, see the usage pattern below.

```{toctree}
---
maxdepth: 1
---
usage_pattern.md
```

## Modules

Notebooks with usage of these components can be found below.

```{toctree}
---
maxdepth: 1
---
modules.md
```