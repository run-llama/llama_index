# Evaluating Evaluators with `LabelledEvaluatorDataset`'s

The purpose of the llama-datasets is to provide builders the means to quickly benchmark
LLM systems or tasks. In that spirit, the `LabelledEvaluatorDataset` exists to
facilitate the evaluation of evaluators in a seamless and effortless manner.

This dataset consists of examples that carries mainly the following attributes:
`query`, `answer`, `ground_truth_answer`, `reference_score`, and `reference_feedback` along with some
other supplementary attributes. The user flow for producing evaluations with this
dataset consists of making predictions over the dataset with a provided LLM
evaluator, and then computing metrics that measure goodness of evaluations by
computationally comparing them to the corresponding references.

Below is a snippet of code that makes use of the `EvaluatorBenchmarkerPack` to
conveniently handle the above mentioned process flow.

```python
from llama_index.llama_dataset import download_llama_dataset
from llama_index.llama_pack import download_llama_pack
from llama_index.evaluation import CorrectnessEvaluator
from llama_index.llms import Gemini
from llama_index import ServiceContext

# download dataset
evaluator_dataset, _ = download_llama_dataset(
    "MiniMtBenchSingleGradingDataset", "./mini_mt_bench_data"
)

# define evaluator
gemini_pro_context = ServiceContext.from_defaults(
    llm=Gemini(model="models/gemini-pro", temperature=0)
)
evaluator = CorrectnessEvaluator(service_context=gemini_pro_context)

# download EvaluatorBenchmarkerPack and define the benchmarker
EvaluatorBenchmarkerPack = download_llama_pack(
    "EvaluatorBenchmarkerPack", "./pack"
)
evaluator_benchmarker = EvaluatorBenchmarkerPack(
    evaluator=evaluators["gpt-3.5"],
    eval_dataset=evaluator_dataset,
    show_progress=True,
)

# produce the benchmark result
benchmark_df = await evaluator_benchmarker.arun(
    batch_size=5, sleep_time_in_seconds=0.5
)
```

## The related `LabelledPairwiseEvaluatorDataset`

A related llama-dataset is the `LabelledPairwiseEvaluatorDataset`, which again
is meant to evaluate an evaluator, but this time where the evaluator is tasked on
comparing a pair of LLM responses to a given query and to determine the better one
amongst them. The usage flow described above is exactly the same as it is for the
`LabelledEvaluatorDataset`, with the exception that the LLM evaluator must be
equipped to perform the pairwise evaluation task — i.e., should be a `PairwiseComparisonEvaluator`.

## More learning materials

To see these datasets in action, be sure to checkout the notebooks listed below
that benchmark LLM evaluators on slightly adapted versions of the MT-Bench dataset.

```{toctree}
---
maxdepth: 1
---

/examples/evaluation/mt_bench_single_grading.ipynb
/examples/evaluation/mt_bench_human_judgement.ipynb
```
