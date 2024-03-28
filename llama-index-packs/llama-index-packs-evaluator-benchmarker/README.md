# Evaluator Benchmarker Pack

A pack for quick computation of benchmark results of your own LLM evaluator
on an Evaluation llama-dataset. Specifically, this pack supports benchmarking
an appropriate evaluator on the following llama-datasets:

- `LabelledEvaluatorDataset` for single-grading evaluations
- `LabelledPairwiseEvaluatorDataset` for pairwise-grading evaluations

These llama-datasets can be downloaed from [llama-hub](https://llamahub.ai).

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack EvaluatorBenchmarkerPack --download-dir ./evaluator_benchmarker_pack
```

You can then inspect the files at `./evaluator_benchmarker_pack` and use them as a template for your own project!

## Code Usage

You can download the pack to the `./evaluator_benchmarker_pack` directory through python
code as well. The sample script below demonstrates how to construct `EvaluatorBenchmarkerPack`
using a `LabelledPairwiseEvaluatorDataset` downloaded from `llama-hub` and a
`PairwiseComparisonEvaluator` that uses GPT-4 as the LLM. Note though that this pack
can also be used on a `LabelledEvaluatorDataset` with a `BaseEvaluator` that performs
single-grading evaluation — in this case, the usage flow remains the same.

```python
from llama_index.core.llama_dataset import download_llama_dataset
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core.evaluation import PairwiseComparisonEvaluator
from llama_index.llms.openai import OpenAI
from llama_index.core import ServiceContext

# download a LabelledRagDataset from llama-hub
pairwise_dataset = download_llama_dataset(
    "MiniMtBenchHumanJudgementDataset", "./data"
)

# define your evaluator
gpt_4_context = ServiceContext.from_defaults(
    llm=OpenAI(temperature=0, model="gpt-4"),
)
evaluator = PairwiseComparisonEvaluator(service_context=gpt_4_context)


# download and install dependencies
EvaluatorBenchmarkerPack = download_llama_pack(
    "EvaluatorBenchmarkerPack", "./pack"
)

# construction requires an evaluator and an eval_dataset
evaluator_benchmarker_pack = EvaluatorBenchmarkerPack(
    evaluator=evaluator,
    eval_dataset=pairwise_dataset,
    show_progress=True,
)

# PERFORM EVALUATION
benchmark_df = evaluator_benchmarker_pack.run()  # async arun() also supported
print(benchmark_df)
```

`Output:`

```text
number_examples                1689
inconclusives                  140
ties                           379
agreement_rate_with_ties       0.657844
agreement_rate_without_ties    0.828205
```

Note that `evaluator_benchmarker_pack.run()` will also save the `benchmark_df` files in the same directory.

```bash
.
└── benchmark.csv
```
