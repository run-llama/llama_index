import os
from llama_index.core.llama_dataset.simple import LabelledSimpleDataset
from pathlib import Path


def main():
    # load synthetic symptom2disease dataset
    synthetic_dataset_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "synthetic_dataset.json",
    )
    synthetic_dataset = LabelledSimpleDataset.from_json(synthetic_dataset_path)

    # get labels
    labels = list(set(ex.reference_label for ex in synthetic_dataset[:]))

    # split labels
    contributor1_labels = labels[::2]
    contributor2_labels = labels[1::2]

    # create splits
    contributor1_examples = [
        ex
        for ex in synthetic_dataset.examples
        if ex.reference_label in contributor1_labels
    ]
    contributor2_examples = [
        ex
        for ex in synthetic_dataset.examples
        if ex.reference_label in contributor2_labels
    ]

    # create LabelledSimpleDataset and save as json
    contributor1_data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "contributor-1",
        "data",
        "contributor1_synthetic_dataset.json",
    )
    Path(contributor1_data_path).parent.absolute().mkdir(parents=True, exist_ok=True)
    contributor2_data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "contributor-2",
        "data",
        "contributor2_synthetic_dataset.json",
    )
    Path(contributor2_data_path).parent.absolute().mkdir(parents=True, exist_ok=True)
    LabelledSimpleDataset(examples=contributor1_examples).save_json(
        contributor1_data_path
    )
    LabelledSimpleDataset(examples=contributor2_examples).save_json(
        contributor2_data_path
    )


if __name__ == "__main__":
    main()
