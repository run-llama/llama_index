# End-to-End Evaluation
End-to-End evaluation should be the guiding signal for your RAG application - will my pipeline generate the right responses given the data sources and a set of queries?

Generally, end-to-end evaluation should be the gold signal for how your application behaves as a whole. While it helps initially to individually inspect queries and responses, as you deal with more failure and corner cases, it may stop being feasible to look at each query individually, and rather it may help instead to define and gain an intuition for what a set of summary metrics or automated evaluation might be telling you.

## Setting up an Evaluation Set

It is helpful to start off with a small but diverse set of queries, and build up more examples as one discovers problematic queries or interactions.

We've created some tools that automatically generate a dataset for you given a set of documents to query. (See example below).


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

Sensitivity testing can be a good inroad into choosing which components to individually test or tweak more thoroughly, or which parts of your dataset (e.g. queries) may be producing problematic results.

More details on how to discover issues automatically with methods such as sensitivity testing will come soon.

Examples of this in the more traditional ML domain include [Giskard](https://docs.giskard.ai/en/latest/getting-started/quickstart.html).

## Metrics Ensembling

It may be expensive to use GPT-4 to carry out evaluation especially as your dev set grows large.

Metrics ensembling uses an ensemble of weaker signals (exact match, F1, ROUGE, BLEU, BERT-NLI and BERT-similarity) to predict the output of a more expensive evaluation methods that are closer to the gold labels (human-labelled/GPT-4).

It is intenteded for two purposes:

1. Evaluating changes cheaply and quickly across a large dataset during the development stage.
2. Flagging outliers for further evaluation (GPT-4 / human alerting) during the production monitoring stage.

We also want the metrics ensembling to be interpretable - the correlation and weighting scores should give an indication of which metrics best capture the  

We will discuss more about the methodology in future updates.
