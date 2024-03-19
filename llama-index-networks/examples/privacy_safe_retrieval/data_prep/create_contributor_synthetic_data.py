import os
from llama_index.core.llama_dataset.simple import LabelledSimpleDataset
from pathlib import Path


import os
from llama_index.core.download.utils import get_file_content
from pathlib import Path


DATA_LINKS = {
    "sigma-1.5": {
        "synthetic": "https://www.dropbox.com/scl/fi/9x3n7q2bnc7tnzxqunx6e/synthetic_dataset.json?rlkey=5oxpuslx97js6u558r6ol5hwb&dl=1",
        "test": "https://www.dropbox.com/scl/fi/ur0uy01m5lw6b593wsps9/symptom_2_disease_test.json?rlkey=uxl5fhi91bhxnzi35d94fu8kl&dl=1",
    },
    "sigma-0.5": {
        "synthetic": "https://www.dropbox.com/scl/fi/wdm1wk591dm6s573ira21/synthetic_dataset.json?rlkey=rttvdktfantkumpao15kg05r1&dl=1",
        "test": "https://www.dropbox.com/scl/fi/3vctf8pbzt66uow5c1cmf/symptom_2_disease_test.json?rlkey=jsmt30tpkffo29tua8t68z8nk&dl=1",
    },
}

EPSILON_VALUES = {"sigma-1.5": 1.3, "sigma-0.5": 15.9}
SIGMA = "sigma-1.5"


def download_dataset_from_dropbox(url: str, local_file_path: str):
    file_raw_content, _ = get_file_content(url, "")
    Path(local_file_path).parent.absolute().mkdir(parents=True, exist_ok=True)
    with open(local_file_path, "w") as f:
        f.write(file_raw_content)


def main():
    # download datasets from dropbox
    synthetic_data_local_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "synthetic_dataset.json"
    )
    download_dataset_from_dropbox(
        DATA_LINKS[SIGMA]["synthetic"], synthetic_data_local_file_path
    )

    test_set_local_file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "symptom_2_disease_test.json",
    )
    download_dataset_from_dropbox(DATA_LINKS[SIGMA]["test"], test_set_local_file_path)

    # load synthetic symptom2disease dataset
    synthetic_dataset = LabelledSimpleDataset.from_json(synthetic_data_local_file_path)

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
