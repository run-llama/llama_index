# Symptom2Disease Example

In this example, we aim to create a privacy-safe, synthetic version of the
Symptom2Disease dataset. We do this by utilizing the `DiffPrivateSimpleDatasetPack`.

## The Symptom2Disease Dataset

The original dataset that is being used for this example demo comes from Kaggle.
([original source](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease)).

To make this dataset work with the `DiffPrivateSimpleDatasetPack`, we first need
to turn it into a `LabelledSimpleDataset`. The `_create_symptom_2_disease_simple_dataset.py`
Python script handles this.

## Creating The Privacy-Safe Synthetic Dataset

To create the synthetic dataset, one ultimately needs to run the `main.py` script.
In order to run the script, however, there are a couple of requirements.

### Requirements

1. Virtual environment setup.

```sh
pyenv virtualenv demo
pyenv activate demo
pip install llama-index llama-index-packs-diff-private-simple-dataset
```

2. Set the ENV variable OPENAI_API_KEY.

The demo uses OpenAI LLMs (that utilize the completion API).

```sh
export OPENAI_API_KEY=...
```

### Running the script

```sh
cd examples
python -m symptom_2_disease.main
```

## Output

After running `main.py`, you will be left with `synthetic_dataset.json`, which
is a JSON format of the `LabelledSimpleDataset`.

```python
from llama_index.core.llama_dataset.simple import LabelledSimpleDataset

synthetic_dataset = LabelledSimpleDataset.from_json("synthetic_dataset.json")
```
