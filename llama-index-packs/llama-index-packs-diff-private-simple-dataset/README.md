# LlamaIndex Packs: `DiffPrivateSimpleDatasetPack`

The `DiffPrivateSimpleDatasetPack` llama pack creates differentially private synthetic
examples from an original, sensitive dataset.

Differential Privacy is a privacy preserving technique that obscures source data
while preserving original attributes, while minimizing the performance impact on
processes that consume the data.

The main motivation for this pack is thus to provide the means to create privacy
safe versions of datasets that can be used in subsequent downstream processing
(i.e., in a prompt to be passed to an LLM) steps. As noted in the original paper (linked below), the synthetic observations can be used as many times as one desires without any additional privacy costs!

The paper appeared at ICLR 2024 and is entitled:
[PRIVACY-PRESERVING IN-CONTEXT LEARNING WITH DIFFERENTIALLY PRIVATE FEW-SHOT GENERATION](https://openreview.net/pdf?id=oZtt0pRnOl).

## How it works?

The pack operates on a dataset represented with the `LabelledSimpleDataset` type.
This type consists of examples called `LabelledSimpleDataExample`, which is a data
class that contains two fields, namely: `text` and `reference_label`. For example,
a news dataset may have example `text`s with `reference_labels` belonging to
`{"World", "Business", "Sports", etc.}`.

The output of this pack's `run()` (and `arun()`) method is another `LabelledSimpleDataset`,
but represents privacy-safe, synthetically generated examples.

## Supported LLMs

To use this pack, an LLM that produces `LogProbs` must be used as it is used in
the differential-privacy generation logic for the next token. The demos found in
the `examples` folder use `OpenAI` completion LLMs (chat completion LLMs were
also used, but these did not produce quality results.)

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack DiffPrivateSimpleDatasetPack --download-dir ./pack
```

You can then inspect the files at `./pack` and use them as a template for your own project!

## Code Usage

You can download the pack from PyPi and then use it your llama-index applications.

```
pip install llama-index-packs-diff-private-simple-dataset
```

A DiffPrivateSimpleDatasetPack object is constructed with the following params:

1. an `LLM` (must return `CompletionResponse`),
2. its associated `tokenizer`,
3. a `PromptBundle` object that contains the parameters required for prompting the LLM to produce the synthetic observations
4. a `LabelledSimpleDataset`
5. [Optional] `sephamore_counter_size` used to help reduce chances of experiencing a `RateLimitError` when calling the LLM's completions API.
6. [Optional] `sleep_time_in_seconds` used to help reduce chances of experiencing a `RateLimitError` when calling the LLM"s completions API.

```python
from llama_index.packs.diff_private_simple_dataset import (
    DiffPrivateSimpleDatasetPack,
)
from llama_index.packs.diff_private_simple_dataset.base import PromptBundle

llm = ...
tokenizer = ...
prompt = PromptBundle(instruction=..., text_heading=..., label_heading=...)

dp_simple_dataset_pack = DiffPrivateSimpleDatasetPack(
    llm=llm,
    tokenizer=tokenizer,
    prompt_bundle=prompt_bundle,
    simple_dataset=simple_dataset,
)
```

If you would like to customize this pack further, then you can download it as a
template:

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
DiffPrivateSimpleDatasetPack = download_llama_pack(
    "DiffPrivateSimpleDatasetPack", "./dense_pack"
)

dp_simple_dataset_pack = DiffPrivateSimpleDatasetPack(
    llm=llm,
    tokenizer=tokenizer,
    prompt_bundle=prompt_bundle,
    simple_dataset=simple_dataset,
)
```

The `run()` function is a light wrapper around `query_engine.query()`. A few
params are required:

- `t_max`: The max number of tokens you would like to generate (the algorithm adds some logic per token in order to satisfy differential privacy).
- `sigma`: Controls the variance of the noise distribution associated with differential privacy noise mechanism. A value of `sigma` amounts to a level of `epsilon` satisfied in differential privacy.
- `num_splits`: The differential privacy algorithm implemented here relies on disjoint splits of the original dataset.
- `num_samples_per_split`: The number of private, in-context examples to include in the generation of the synthetic example.

```python
synthetic_dataset = dp_simple_dataset_pack.run(
    sizes={"World": 1, "Sports": 1, "Sci/Tech": 0, "Business": 0},
    t_max=10,  #
    sigma=0.5,
    num_splits=2,
    num_samples_per_split=8,
)

print(response)
```

## Examples

- See [examples/basic_demo](https://github.com/run-llama/llama_index/tree/main/llama-index-packs/llama-index-packs-diff-private-simple-dataset/examples/basic_demo) folder for a notebook the consists of a basic demo
  on how to use the `DiffPrivateSimpleDatasetPack`.
- Also see [examples/symptom_2_disease](https://github.com/run-llama/llama_index/tree/main/llama-index-packs/llama-index-packs-diff-private-simple-dataset/examples/symptom_2_disease) for a more Python program that generates
  a synthetic version of the [Symptom2Disease](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease) dataset.
