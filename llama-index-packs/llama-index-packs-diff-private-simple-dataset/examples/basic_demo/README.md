# Basic Demo

For this basic demo, we illustrate how to us the `DiffPrivateSimpleDatasetPack`
to create a privacy-safe, synthetic version of a supplied dataset, namely
the AGNews classification dataset.

## The AGNews Dataset

The original dataset that is being used for this basic demo comes from Kaggle.
([original source](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)).

To make this dataset work with the `DiffPrivateSimpleDatasetPack`, we first need
to turn it into a `LabelledSimpleDataset`. The `_create_agnews_simple_dataset.ipynb`
notebook does this exactly.

## The Demo Notebook

The `demo_usage.ipynb` notebook illustrates how to create privacy-safe, synthetic
examples of the AGNews (in `LabelledSimpleDataset` format) dataset.
