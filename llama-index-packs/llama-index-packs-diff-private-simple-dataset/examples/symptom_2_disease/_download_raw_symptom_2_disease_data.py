"""Download the raw Symptom2Disease csv file."""

import os
from llama_index.core.download.utils import get_file_content
from pathlib import Path


DATASET_DROPBOX_LINK = "https://www.dropbox.com/scl/fi/2edfoyaimxds21yylqqpv/Symptom2Disease.csv?rlkey=1pj586zloa7np3klgxn7z1qwh&dl=1"


def download_dataset_from_dropbox():
    file_raw_content, _ = get_file_content(DATASET_DROPBOX_LINK, "")
    local_source_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "Symptom2Disease.csv"
    )
    Path(local_source_file_path).parent.absolute().mkdir(parents=True, exist_ok=True)
    with open(local_source_file_path, "w") as f:
        f.write(file_raw_content)


if __name__ == "__main__":
    download_dataset_from_dropbox()
