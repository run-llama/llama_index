# Evaluation

## Setting the Stage

LlamaIndex is meant to connect your data to your LLM applications.

Sometimes, even after diagnosing and fixing bugs by looking at traces, more fine-grained evaluation is required to systematically diagnose issues.

LlamaIndex aims to provide those tools to make identifying issues and receiving useful diagnostic signals easy.

Closely tied to evaluation are the concepts of experimentation and experiment tracking.

```{toctree}
---
maxdepth: 1
---
/end_to_end_tutorials/dev_practices/component_wise_evaluation.md
/end_to_end_tutorials/dev_practices/e2e_evaluation.md
```

## Diving Deeper into Evaluation
Evaluation is a controversial topic, and as the field of NLP has evolved, so have the methods of evaluation.

In a world where powerful foundation models are now performing annotation tasks better than human annotators, the best practices around evaluation are constantly changing. Previous methods of evaluation which were used to bootstrap and evaluate today's models such as BLEU or F1 have been shown to have poor correlation with human judgements, and need to be applied prudently.

Typically, generation-heavy, open-ended tasks and requiring judgement or opinion and harder to evaluate automatically than factual questions due to their subjective nature.

## Resources
1. [LLM Patterns by Eugene Yan](https://eugeneyan.com/writing/llm-patterns/)

