(Evaluation)=
# Evaluation

## Setting the Stage

LlamaIndex is meant to connect your data to your LLM applications.

Sometimes, even after diagnosing and fixing bugs by looking at traces, more fine-grained evaluation is required to systematically diagnose issues.

LlamaIndex aims to provide those tools to make identifying issues and receiving useful diagnostic signals easy.

Closely tied to evaluation are the concepts of experimentation and experiment tracking.

## General Strategy

When developing your LLM application, it could help to first define an end-to-end evaluation workflow, and then once you've started collecting failure or corner cases and getting an intuition for what is or isn't going well, you may dive deeper into evaluating and improving specific components. 

The analogy with software testing is integration tests and unit tests. You should probably start writing unit tests once you start fiddling with individual components. Equally, your gold standard on whether things are working will together are integration tests. Both are equally important.

### E2E or Component-Wise - Which Do I Start With?
However, if you want to get an overall idea of how your system is doing as you iterate upon it, it makes sense to start with centering your core development loop around the e2e eval - as an overall sanity/vibe check.

If you have an idea of what you're doing and want to iterate step by step on each component, building it up as things go - you may want to start with a component-wise eval. However this may run the risk of premature optimization - making model selection or parameter choices without assessing the overall application needs. You may have to revisit these choices after creating your final application.

```{toctree}
---
maxdepth: 1
---
/end_to_end_tutorials/dev_practices/e2e_evaluation.md
/end_to_end_tutorials/dev_practices/component_wise_evaluation.md
```

Here is an overview of the existing modules for evaluation. We will be adding more modules and support over time.

```{toctree}
---
maxdepth: 1
---
/core_modules/supporting_modules/evaluation/root.md
```


## Diving Deeper into Evaluation
Evaluation is a controversial topic, and as the field of NLP has evolved, so have the methods of evaluation.

In a world where powerful foundation models are now performing annotation tasks better than human annotators, the best practices around evaluation are constantly changing. Previous methods of evaluation which were used to bootstrap and evaluate today's models such as BLEU or F1 have been shown to have poor correlation with human judgements, and need to be applied prudently.

Typically, generation-heavy, open-ended tasks and requiring judgement or opinion and harder to evaluate automatically than factual questions due to their subjective nature. We will aim to provide more guides and case-studies for which methods are appropriate in a given scenario.

### Standard Metrics

Against annotated datasets, whether your own data or an academic benchmark, there are a number of standard metrics that it helps to be aware of:

1. **Exact Match (EM):** The percentage of queries that are answered exactly correctly.
2. **F1:** The percentage of queries that are answered exactly correctly or with a small edit distance (e.g. 1-2 words).
3. **Recall:** The percentage of queries that are answered correctly, regardless of the number of answers returned.
4. **Precision:** The percentage of queries that are answered correctly, divided by the number of answers returned.

This [towardsdatascience article](https://towardsdatascience.com/ranking-evaluation-metrics-for-recommender-systems-263d0a66ef54) covers more technical metrics like NDCG, MAP and MRR in greater depth.

## Resources
1. [LLM Patterns by Eugene Yan](https://eugeneyan.com/writing/llm-patterns/)

