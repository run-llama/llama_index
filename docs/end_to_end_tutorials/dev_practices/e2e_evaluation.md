# End-to-End Evaluation
## Setting up an Evaluation Set

To help with this, we've created some tools that automatically generate a dataset for you given a set of documents to query.

```{toctree}
---
maxdepth: 1
---
/examples/evaluation/QuestionGeneration.ipynb
```

In the future, we will also be able to create datasets automatically against tools.

## Qualitative v.s. Quantitative Eval

Quantitative eval is more useful when evaluating applications where there is a correct answer - for instance, validating that the choice of tools and their inputs are correct given the plan, or retrieving specific pieces of information, or attempting to produce intermediate output of a certain schema (e.g. JSON fields).

Qualitative eval is more useful when generating long-form responses that are meant to be *helpful* but not necessarily completely accurate.


## Discovery - Sensitivity Testing

With a complex pipeline, it may be unclear which parts of the pipeline are affecting your results.

Sensitivity testing can be a good inroad into choosing which components to individually test or tweak more thoroughly.

More details on how to discover issues automatically with methods such as sensitivity testing will come soon.

## Metrics Ensembling

It may be expensive to use GPT-4 to carry out evaluation especially as your dev set grows large.

Metrics ensembling uses an ensemble of weaker signals (exact match, F1, ROUGE, BLEU, BERT-NLI and BERT-similarity) to predict the output of a more expensive evaluation methods that are closer to the gold labels (human-labelled/GPT-4).

It is intenteded for two purposes:

1. Evaluating changes cheaply and quickly across a large dataset during the development stage.
2. Flagging outliers for further evaluation (GPT-4 / human alerting) during the production monitoring stage.

We also want the metrics ensembling to be interpretable - the correlation and weighting scores should give an indication of which metrics best capture the  

We will discuss more about the methodology in future updates.