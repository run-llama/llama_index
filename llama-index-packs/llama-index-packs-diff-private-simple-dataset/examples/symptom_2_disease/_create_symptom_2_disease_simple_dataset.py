"""Creates a `LabelledSimpleDataset` version of the Symptom2Dataset."""

import os
import pandas as pd
from llama_index.core.llama_dataset.simple import (
    LabelledSimpleDataExample,
    LabelledSimpleDataset,
)
from llama_index.core.llama_dataset.base import CreatedBy, CreatedByType
from sklearn.model_selection import train_test_split


def create_labelled_simple_dataset_from_df(df: pd.DataFrame) -> LabelledSimpleDataset:
    examples = []
    for index, row in df.iterrows():
        example = LabelledSimpleDataExample(
            reference_label=row["label"],
            text=row["text"],
            text_by=CreatedBy(type=CreatedByType.HUMAN),
        )
        examples.append(example)

    return LabelledSimpleDataset(examples=examples)


def main():
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "Symptom2Disease.csv"
    )
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        raise ValueError(
            "The `Symptom2Disease.csv` file cannot be found. Please "
            "run the `_download_raw_symptom_2_disease_data.py` script."
        )
    train, test = train_test_split(df, test_size=0.2)

    train_simple_dataset = create_labelled_simple_dataset_from_df(train)
    test_simple_dataset = create_labelled_simple_dataset_from_df(test)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    train_simple_dataset.save_json(os.path.join(output_path, "symptom_2_disease.json"))
    test_simple_dataset.save_json(
        os.path.join(output_path, "symptom_2_disease_test.json")
    )


if __name__ == "__main__":
    main()
